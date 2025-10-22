# -*- coding: utf-8 -*-
"""
saasdj-backend (open v2)

- FastAPI para analisar .mp3/.wav e:
  - Estimar BPM de forma robusta (anti-half/double, prior 118–138)
  - Extrair bandas espectrais (low/mid/high), HP ratio, onset, kickband e duração
  - Pedir ao GPT uma classificação ABERTA (sem lista fixa):
      subgênero principal + 1 influência + breve explicação
- Resposta SEMPRE inclui:
  {
    "bpm": <int|None>,
    "subgenero": <str>,
    "influencia": <str>,
    "explicacao": <str>,
    "analise": "<linha técnica>"
  }

Requisitos mínimos (pip):
fastapi uvicorn librosa soundfile pydub scipy requests python-multipart
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import requests
import librosa
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# =============================================================================
# Configuração
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")


# =============================================================================
# Leitura de áudio (janela mid90)
# =============================================================================

def _load_segment(file_bytes: bytes, offset_seconds: float, duration_seconds: float, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Carrega um segmento específico (offset/duration) com fallback pydub->wav."""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True,
                             offset=offset_seconds, duration=duration_seconds)
        if y is None or len(y) == 0:
            raise ValueError("Áudio vazio após leitura principal")
        return y, sr
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        start_ms = int(offset_seconds * 1000.0)
        end_ms = int((offset_seconds + duration_seconds) * 1000.0)
        segment = audio[start_ms:end_ms]
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr


def _measure_duration_sec(file_bytes: bytes) -> float | None:
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        return len(audio) / 1000.0
    except Exception:
        return None


def load_audio_window_mid90(file_bytes: bytes, sr: int = 22050) -> Tuple[np.ndarray, int, float]:
    """
    Janela única 'mid90': 90s a partir de 60s (1:00 -> 2:30).
    Fallbacks:
      - se a faixa < 150s: usa 90s centrados (ou o máximo possível).
    Retorna (y, sr, duration_sec_total).
    """
    duration_sec = _measure_duration_sec(file_bytes)

    # alvo: offset=60.0, dur=90.0
    target_off, target_dur = 60.0, 90.0

    if duration_sec is None:
        # sem metadados de duração — tenta carregar a partir de 60s por 90s
        y, sr = _load_segment(file_bytes, target_off, target_dur, sr)
        return y, sr, float(len(y) / sr)

    if duration_sec >= 150.0:
        off = target_off
        dur = min(target_dur, duration_sec - off)
    else:
        # se não tem 150s, pega 90s centrados, ou o que couber
        center = max(0.0, duration_sec / 2.0 - 45.0)
        off = max(0.0, min(center, max(0.0, duration_sec - 90.0)))
        dur = min(90.0, duration_sec - off)

    y, sr = _load_segment(file_bytes, off, dur, sr)
    return y, sr, float(duration_sec)


# =============================================================================
# Utilidades de áudio / features
# =============================================================================

def _bands_energy(y: np.ndarray, sr: int) -> Tuple[float, float, float, float]:
    """Energia em Low/Mid/High + subbanda de kick (40–100Hz)."""
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1 / sr)

    low = float(yf[(xf >= 20) & (xf < 250)].sum())
    mid = float(yf[(xf >= 250) & (xf < 4000)].sum())
    high = float(yf[(xf >= 4000) & (xf <= 20000)].sum())
    kick = float(yf[(xf >= 40) & (xf < 100)].sum())

    return low, mid, high, kick


# =============================================================================
# BPM robusto (anti half/double + prior 4x4)
# =============================================================================

def _tempo_via_beattrack(y: np.ndarray, sr: int) -> List[float]:
    cands = []
    try:
        try:
            _, P = librosa.effects.hpss(y)
            y_bt = P if P is not None and len(P) else y
        except Exception:
            y_bt = y
        t = librosa.beat.tempo(y=y_bt, sr=sr, aggregate=None, max_tempo=200.0)
        if t is not None and len(t):
            cands += list(np.asarray(t, dtype=float))
    except Exception:
        pass
    return cands

def _tempo_via_onset(y: np.ndarray, sr: int) -> List[float]:
    cands = []
    try:
        _, P = librosa.effects.hpss(y)
        y_src = P if P is not None and len(P) else y
        onset_env = librosa.onset.onset_strength(y=y_src, sr=sr)
        if onset_env is not None and len(onset_env):
            t = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None, max_tempo=200.0)
            if t is not None and len(t):
                cands += list(np.asarray(t, dtype=float))
    except Exception:
        pass
    return cands

def _add_half_double(cands: List[float]) -> List[float]:
    out = []
    for b in cands:
        if b <= 0 or np.isnan(b) or np.isinf(b):
            continue
        for k in (0.5, 1.0, 2.0):
            v = b * k
            if 60.0 <= v <= 200.0:
                out.append(v)
    return out

def estimate_bpm_robust(y: np.ndarray, sr: int, kick_val: float | None, onset_strength: float | None) -> float | None:
    """
    Estima BPM combinando múltiplos métodos + heurísticas anti metade/dobro.
    - Junta candidatos de beat_track e onset/tempo.
    - Gera variações half/double.
    - Aplica 'prior' suave para 118–138 (4x4), e 122–134 (main groove).
    - Corrige meio/dobro com base na distribuição dos candidatos.
    """
    cands = []
    cands += _tempo_via_beattrack(y, sr)
    cands += _tempo_via_onset(y, sr)
    if not cands:
        return None

    cands = _add_half_double(cands)

    # cluster por arredondamento
    rounded = np.round(cands).astype(int)
    uniq, counts = np.unique(rounded, return_counts=True)
    scores = {int(u): float(c) for u, c in zip(uniq, counts)}  # pontuação base: frequência

    # prior suave para 4x4
    for u in list(scores.keys()):
        if 118 <= u <= 138:
            scores[u] *= 1.15
        if 122 <= u <= 134:
            scores[u] *= 1.05
        # empurrãozinho quando bem percussivo
        if (onset_strength or 0.0) >= 0.60:
            scores[u] *= 1.03

    # escolhe melhor cluster
    cand = max(scores.items(), key=lambda kv: kv[1])[0]
    bpm = float(cand)

    # anti-dobro: saiu <110 mas há apoio forte em ~2x?
    if bpm < 110:
        double = bpm * 2.0
        around_double = [x for x in rounded if abs(x - int(round(double))) <= 2]
        if len(around_double) >= max(2, len([x for x in rounded if abs(x - int(round(bpm))) <= 2])):
            bpm = double

    # anti-metade: saiu >170 e há apoio em ~/2?
    if bpm > 170:
        half = bpm / 2.0
        around_half = [x for x in rounded if abs(x - int(round(half))) <= 2]
        if len(around_half) >= max(2, len([x for x in rounded if abs(x - int(round(bpm))) <= 2])):
            bpm = half

    if bpm < 60 or bpm > 200:
        return None
    return float(np.clip(bpm, 60.0, 200.0))


def extract_features(y: np.ndarray, sr: int, duration_sec: float) -> Dict[str, float | int | None]:
    """Features com BPM robusto (anti half/double) + duração."""
    # Espectro (inclui 'kick' que também ajuda indiretamente o BPM)
    low, mid, high, kick = _bands_energy(y, sr)
    total = max(low + mid + high, 1e-9)
    low_pct = float(low / total)
    mid_pct = float(mid / total)
    high_pct = float(high / total)

    # HPSS e onset normalizado
    try:
        H, P = librosa.effects.hpss(y)
        hp_ratio = float((np.mean(np.abs(H)) + 1e-8) / (np.mean(np.abs(P)) + 1e-8))
    except Exception:
        hp_ratio = 1.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = float(np.mean(onset_env) / (np.std(onset_env) + 1e-6)) if onset_env is not None and len(onset_env) else 0.0
    except Exception:
        onset_strength = 0.0

    # BPM robusto
    bpm = estimate_bpm_robust(y, sr, kick_val=kick, onset_strength=onset_strength)

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "energy_low": low,
        "energy_mid": mid,
        "energy_high": high,
        "kick_40_100": kick,
        "low_pct": round(low_pct, 6),
        "mid_pct": round(mid_pct, 6),
        "high_pct": round(high_pct, 6),
        "hp_ratio": round(hp_ratio, 6),
        "onset_strength": round(onset_strength, 6),
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
    }


# === Helper: linha técnica na resposta ===
def _fmt_pct(x: float | None) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"

def _fmt_float(x: float | None, nd=2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"

def build_tech_line(
    feats: Dict[str, float | int | None],
    chosen_sub: str,
    decision_source: str,
) -> str:
    bpm = feats.get("bpm")
    low_pct = feats.get("low_pct")
    mid_pct = feats.get("mid_pct")
    high_pct = feats.get("high_pct")
    hp = feats.get("hp_ratio")
    onset = feats.get("onset_strength")
    kick = feats.get("kick_40_100")
    dur = feats.get("duration_sec")

    parts = [
        f"BPM={int(round(bpm)) if bpm is not None else 'n/a'}",
        f"low%={_fmt_pct(low_pct)}",
        f"mid%={_fmt_pct(mid_pct)}",
        f"high%={_fmt_pct(high_pct)}",
        f"hp={_fmt_float(hp,2)}",
        f"onset={_fmt_float(onset,2)}",
        f"kick={int(kick) if kick is not None else 'n/a'}",
        f"dur={int(dur) if dur is not None else 'n/a'}s",
        f"chosen={chosen_sub}",
        f"source={decision_source}",
    ]
    return "; ".join(parts)


# =============================================================================
# LLM (classificação aberta)
# =============================================================================

PROMPT_OPEN = """
Você é um A&R especializado em música eletrônica. Receberá FEATURES de uma faixa:
- BPM estimado (robusto anti half/double)
- distribuição espectral low/mid/high (percentuais)
- hp_ratio (harmônico ÷ percussivo)
- onset_strength (força média de transientes)
- energia de kick (40–100Hz)
- duração total em segundos
A janela analisada é de ~90s entre 60–150s (1:00–2:30), considerada o trecho mais representativo.

TAREFA:
1) Dê um SUBGÊNERO principal plausível (sem lista fixa; use a taxonomia que fizer sentido, p.ex. "Tech House", "Melodic House & Techno", "Indie Dance", "Progressive House", "Peak Time Techno", "Bass House", "Afro House", "Deep House", "Big Room", "Progressive EDM", "Psytrance", "Drum & Bass", etc. Evite rótulos fora do domínio eletrônico quando os dados indicarem EDM 4x4).
2) Indique 1 INFLUÊNCIA (outro subgênero/vertente que aparece como caráter secundário).
3) Explique brevemente o porquê (máx. 3–4 linhas), referindo-se a BPM, brilho (high%), presença melódica (hp_ratio), pegada de pista (kick/onsets) e duração.

FORMATO DE SAÍDA (JSON **válido**):
{
  "subgenero": "<string>",
  "influencia": "<string>",
  "explicacao": "<string concisa>"
}

REGRAS:
- Seja coerente com o BPM: se vier ~120–136 com 4x4 evidente e brilho/hi-hat moderado, prefira nomes da família House/Techno em vez de rotular como Trip Hop/Hip Hop.
- Se a faixa for longa (>= 300s) e mais melódica (hp_ratio alto, mid% forte), Progressive House e/ou Melodic House & Techno são opções frequentes.
- Se muito brilhante (high% elevado) e com drops nítidos, considerar Progressive EDM/Big Room.
- Se BPM ~128–136 com pegada dura/industrial e high% não tão alto, considerar Peak Time/Hard Techno; se extremamente agressivo e muito rápido, considerar Hard Dance subgêneros.
- Seja sucinto e técnico na justificativa.
"""

def call_gpt_open(features: Dict[str, float | int | None]) -> Dict[str, str]:
    """Envia FEATURES ao LLM e retorna dict com subgenero, influencia, explicacao."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    user_payload = {
        "FEATURES": features
    }
    messages = [
        {"role": "system", "content": PROMPT_OPEN},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=60)
    try:
        body = r.json()
    except Exception:
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - resposta não-JSON: {r.text[:500]}")
    if r.status_code != 200:
        err = body.get("error", {})
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - {err.get('type')} - {err.get('message')}")
    if "choices" not in body or not body["choices"]:
        raise RuntimeError(f"OpenAI resposta inesperada: {body}")

    text = body["choices"][0]["message"]["content"].strip()
    # tenta parsear JSON (o prompt já força JSON válido)
    try:
        parsed = json.loads(text)
        # saneamento básico de chaves
        return {
            "subgenero": str(parsed.get("subgenero", "Subgênero Não Identificado"))[:120],
            "influencia": str(parsed.get("influencia", ""))[:120],
            "explicacao": str(parsed.get("explicacao", ""))[:600],
        }
    except Exception:
        # fallback: retorna tudo no campo explicacao
        return {
            "subgenero": "Subgênero Não Identificado",
            "influencia": "",
            "explicacao": text[:600],
        }


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="saasdj-backend (open v2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "open-v2"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Recebe um .mp3/.wav, extrai features e classifica (aberto).
    Retorno:
    {
      "bpm": <int|None>,
      "subgenero": <str>,
      "influencia": <str>,
      "explicacao": <str>,
      "analise": "BPM=...; low%=...; mid%=...; high%=...; hp=...; onset=...; kick=...; dur=...s; chosen=...; source=llm|fallback"
    }
    """
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 1) Carregar janela mid90 e extrair features
        y, sr, duration_sec = load_audio_window_mid90(data)
        feats = extract_features(y, sr, duration_sec)

        # 2) Chamar GPT (classificação aberta)
        try:
            result = call_gpt_open(feats)
            sub = result.get("subgenero") or "Subgênero Não Identificado"
            infl = result.get("influencia") or ""
            exp = result.get("explicacao") or ""
            decision_source = "llm"
        except Exception as e:
            # Falha na OpenAI — Fallback muito simples: sem taxonomia fixa, logo marcamos como não identificado
            sub = "Subgênero Não Identificado"
            infl = ""
            exp = f"Classificação não disponível (fallback): {e}"
            decision_source = "fallback"

        # 3) BPM como INTEIRO na resposta (arredondado)
        bpm_val = feats.get("bpm")
        bpm_out = int(round(bpm_val)) if bpm_val is not None else None

        # 4) Linha técnica
        tech_line = build_tech_line(
            feats=feats,
            chosen_sub=sub,
            decision_source=decision_source,
        )

        return {
            "bpm": bpm_out,
            "subgenero": sub,
            "influencia": infl,
            "explicacao": exp,
            "analise": tech_line,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        # Erro inesperado
        payload = {
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "influencia": "",
            "explicacao": f"processing failed: {e.__class__.__name__}: {e}",
        }
        try:
            # tenta aproveitar variáveis locais se existirem
            feats = locals().get("feats", None)
            if feats:
                bpm_val = feats.get("bpm")
                bpm_out = int(round(bpm_val)) if bpm_val is not None else None
                tech_line = build_tech_line(
                    feats=feats,
                    chosen_sub="Subgênero Não Identificado",
                    decision_source="fallback-error",
                )
                payload.update({"bpm": bpm_out, "analise": tech_line})
        except Exception:
            pass

        return JSONResponse(status_code=500, content=payload)
