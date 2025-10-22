# -*- coding: utf-8 -*-
"""
saasdj-backend (v6 - análise 100% livre)

- FastAPI para analisar .mp3/.wav
- Extrai features com librosa (BPM, bandas espectrais, HP ratio, onset, kickband, duração)
- NÃO usa lista/taxonomia de subgêneros. O LLM decide livremente:
    - "subgenero"        (string)  -> um subgênero principal
    - "influencia"       (string)  -> exatamente UMA influência de outro subgênero/vertente
    - "explicacao"       (string)  -> breve justificativa (2–4 frases)
    - "bpm"              (int|null)-> pode ajustar o estimado se parecer metade/dobro
- Retorna também uma linha "analise" com os números extraídos (debug)

Requisitos mínimos (pip):
fastapi uvicorn librosa soundfile pydub scipy requests python-multipart
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Tuple, Any

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
# Carregamento da janela (mid90: 1:00 → 2:30)
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


def load_window_mid90(file_bytes: bytes, sr: int = 22050) -> Tuple[Tuple[np.ndarray, int], float | None]:
    """
    Janela 'mid90': 90s a partir de 60s (1:00 -> 2:30).
    Fallback: se faixa < 150s, usa ~90s centrados (ou o máximo possível).
    Retorna ((y, sr), duration_sec)
    """
    duration_sec = _measure_duration_sec(file_bytes)

    target_off, target_dur = 60.0, 90.0

    if duration_sec is None:
        y, sr = _load_segment(file_bytes, target_off, target_dur, sr)
        return (y, sr), float(len(y) / sr)

    if duration_sec >= 150.0:
        off = target_off
        dur = min(target_dur, duration_sec - off)
    else:
        center = max(0.0, duration_sec / 2.0 - 45.0)
        off = max(0.0, min(center, max(0.0, duration_sec - 90.0)))
        dur = min(90.0, duration_sec - off)

    y, sr = _load_segment(file_bytes, off, dur, sr)
    return (y, sr), float(duration_sec)


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


def _tempo_candidates(y: np.ndarray, sr: int) -> List[float]:
    """
    Estimativas de BPM com ênfase percussiva e candidatas em half/double:
    - beat_track no percussivo
    - onset -> tempo no percussivo
    - retorna base, base*0.5, base*2 filtradas 60–200
    """
    bpms: List[float] = []
    try:
        H, P = librosa.effects.hpss(y)
    except Exception:
        H, P = None, None

    # 1) beat_track no percussivo
    try:
        y_for_tempo = P if P is not None and len(P) else y
        tempo_bt, _ = librosa.beat.beat_track(y=y_for_tempo, sr=sr)
        if tempo_bt and tempo_bt > 0:
            bpms.append(float(tempo_bt))
    except Exception:
        pass

    # 2) onset -> tempo no percussivo
    try:
        y_for_onset = P if P is not None and len(P) else y
        onset_env = librosa.onset.onset_strength(y=y_for_onset, sr=sr)
        if onset_env is not None and len(onset_env):
            t = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, max_tempo=200)
            if t is not None and len(t):
                bpms.append(float(np.median(t)))
    except Exception:
        pass

    if not bpms:
        return []

    base = float(np.median(bpms))
    cands = [base * 0.5, base, base * 2.0]
    return [b for b in cands if 60.0 <= b <= 200.0]


def _choose_bpm_from_candidates(bpm_cands: List[float]) -> float | None:
    """Escolhe BPM preferindo valores <=150 quando existir par dobro/metade."""
    if not bpm_cands:
        return None
    arr = np.array(bpm_cands, dtype=float)
    for b in arr:
        if b > 150 and np.any(np.isclose(arr, b / 2.0, atol=2.0)):
            return float(np.median(arr[np.isclose(arr, b / 2.0, atol=2.0)]))
    return float(np.median(arr))


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float | int | None]:
    """Features de UMA janela."""
    bpm_candidates = _tempo_candidates(y, sr)
    bpm = _choose_bpm_from_candidates(bpm_candidates)

    low, mid, high, kick = _bands_energy(y, sr)
    total = max(low + mid + high, 1e-9)
    low_pct = float(low / total)
    mid_pct = float(mid / total)
    high_pct = float(high / total)

    H, P = librosa.effects.hpss(y)
    hp_ratio = float((np.mean(np.abs(H)) + 1e-8) / (np.mean(np.abs(P)) + 1e-8))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = float(np.mean(onset_env) / (np.std(onset_env) + 1e-6)) if (onset_env is not None and len(onset_env)) else 0.0

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
    }


def extract_features_mid90(file_bytes: bytes) -> Dict[str, float | int | None]:
    """Extrai features da janela mid90 e inclui 'duration_sec'."""
    (y, sr), duration_sec = load_window_mid90(file_bytes)
    f = extract_features(y, sr)
    f["duration_sec"] = float(duration_sec) if duration_sec is not None else None
    return f


# =============================================================================
# Helpers de formatação
# =============================================================================

def _fmt_pct(x: float | None) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"

def _fmt_float(x: float | None, nd=2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"

def build_tech_line(
    feats: Dict[str, float | int | None],
    chosen_src: str,
    chosen_sub: str | None = None,
) -> str:
    """Linha compacta com os números extraídos (debug)."""
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
        f"chosen={chosen_sub or 'n/a'}",
        f"source={chosen_src}",
    ]
    return "; ".join(parts)


# =============================================================================
# LLM (JSON estrito): subgênero livre, 1 influência, explicação
# =============================================================================

PROMPT_SYSTEM = (
    "Você é um especialista em música eletrônica. "
    "Receberá FEATURES extraídas de ~90s da faixa (entre 1:00 e 2:30) + duração total. "
    "Analise e responda em JSON estrito."
)

PROMPT_USER_TEMPLATE = """
FEATURES (mid90) e duração total:
{features_json}

Tarefas:
1) Estime/valide o BPM (inteiro). Se parecer metade/dobro incorreto, corrija.
2) Escolha UM subgênero principal (string). Não use termos genéricos como "Electronic" ou "EDM": seja específico (ex.: "Tech House", "Melodic House & Techno", "Indie Dance", "Progressive House", "Hard Techno", "Dubstep", etc.). Pode usar qualquer nome comum na cena eletrônica.
3) Indique UMA influência (string) de outro subgênero/vertente que aparece na faixa.
4) Explique em 2–4 frases de forma concisa por que esse subgênero e essa influência fazem sentido, citando textura (low/mid/high), presença melódica (hp_ratio), impacto (onset/kick) e, se relevante, a duração.

Formato da RESPOSTA (JSON estrito):
{{
  "bpm": <int or null>,
  "subgenero": "<string>",
  "influencia": "<string>",
  "explicacao": "<string concisa>"
}}
"""

def _openai_chat(messages: List[Dict[str, str]], response_json: bool = True) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"model": MODEL, "messages": messages, "temperature": 0}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    try:
        body = r.json()
    except Exception:
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - resposta não-JSON: {r.text[:500]}")
    if r.status_code != 200:
        err = body.get("error", {})
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - {err.get('type')} - {err.get('message')}")
    if "choices" not in body or not body["choices"]:
        raise RuntimeError(f"OpenAI resposta inesperada: {body}")
    return body["choices"][0]["message"]["content"].strip()


def call_gpt_free(features: Dict[str, float | int | None]) -> Dict[str, Any]:
    """
    Envia as features e recebe:
      bpm (int|null), subgenero (str), influencia (str), explicacao (str)
    """
    user_text = PROMPT_USER_TEMPLATE.format(
        features_json=json.dumps(features, ensure_ascii=False, indent=2),
    )
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    content = _openai_chat(messages, response_json=True)
    try:
        data = json.loads(content)
        out: Dict[str, Any] = {
            "bpm": int(data["bpm"]) if data.get("bpm") is not None else None,
            "subgenero": str(data.get("subgenero") or "").strip() or "Subgênero Não Identificado",
            "influencia": str(data.get("influencia") or "").strip(),
            "explicacao": str(data.get("explicacao") or "").strip(),
        }
        return out
    except Exception as e:
        raise RuntimeError(f"Falha ao parsear JSON do LLM: {e}: {content[:2000]}")


# =============================================================================
# Fallback heurístico (se LLM falhar)
# =============================================================================

def heuristic_fallback(features: Dict[str, float | int | None]) -> Dict[str, Any]:
    """
    Fallback simples, só para não quebrar:
    tenta inferir um subgênero plausível e UMA influência a partir de números.
    """
    bpm = features.get("bpm")
    lp = features.get("low_pct") or 0.0
    mp = features.get("mid_pct") or 0.0
    hpv = features.get("high_pct") or 0.0
    hpr = features.get("hp_ratio") or 0.0
    onset = features.get("onset_strength") or 0.0
    dur = features.get("duration_sec") or 0.0

    sub = "Subgênero Não Identificado"
    inf = ""

    if bpm is not None:
        if 120 <= bpm <= 128:
            if hpr >= 1.4 and mp >= 0.45:
                sub = "Melodic House & Techno"; inf = "Progressive House"
            elif dur >= 300 and hpv <= 0.28:
                sub = "Progressive House"; inf = "Melodic House & Techno"
            elif lp >= 0.45:
                sub = "Bass House"; inf = "Tech House"
            else:
                sub = "Tech House"; inf = "Progressive House"
        elif 110 <= bpm <= 123 and mp >= 0.48 and hpv <= 0.32 and hpr >= 1.3:
            sub = "Indie Dance"; inf = "Deep House"
        elif 132 <= bpm <= 140 and (hpr >= 1.2) and (mp >= 0.42):
            sub = "Melodic & Progressive Trance"; inf = "Peak Time Techno"
        elif 135 <= bpm <= 165 and onset >= 0.65:
            sub = "Hard Techno"; inf = "Hard Dance/Groove"
        elif 136 <= bpm <= 146 and lp >= 0.45 and onset >= 0.60:
            sub = "Dubstep"; inf = "Riddim"

    exp = (
        f"BPM ~{int(round(bpm)) if bpm else 'n/a'}, "
        f"bandas low/mid/high≈{lp:.2f}/{mp:.2f}/{hpv:.2f}, "
        f"HP≈{hpr:.2f}, onset≈{onset:.2f}, duração≈{int(dur)}s. "
        "Rótulo e influência inferidos por regras simples."
    )

    return {
        "bpm": int(round(bpm)) if bpm is not None else None,
        "subgenero": sub,
        "influencia": inf,
        "explicacao": exp,
    }


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="saasdj-backend (v6 - análise livre)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v6"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Recebe .mp3/.wav, extrai features da janela mid90 e pede ao LLM:
      - bpm (int|null)
      - subgenero (str livre)
      - influencia (str única)
      - explicacao (str curta)
    Resposta inclui também "analise" (linha técnica de debug).
    """
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 1) Extrai features
        feats = extract_features_mid90(data)

        # 2) LLM livre (sem taxonomia/listas)
        try:
            llm = call_gpt_free(feats)
            sub = llm.get("subgenero") or "Subgênero Não Identificado"
            inf = llm.get("influencia") or ""
            exp = llm.get("explicacao") or ""
            # BPM: usa o do LLM se veio inteiro, senão o estimado
            bpm_out = llm.get("bpm")
            if not isinstance(bpm_out, int):
                bpm_est = feats.get("bpm")
                bpm_out = int(round(bpm_est)) if bpm_est is not None else None

            tech_line = build_tech_line(feats, chosen_src="llm", chosen_sub=sub)
            return {
                "bpm": bpm_out,
                "subgenero": sub,
                "influencia": inf,
                "explicacao": exp,
                "analise": tech_line,
            }

        except Exception as e:
            # 3) Fallback heurístico
            guess = heuristic_fallback(feats)
            tech_line = build_tech_line(feats, chosen_src="fallback", chosen_sub=guess["subgenero"])
            return JSONResponse(
                status_code=502,
                content={
                    "bpm": guess["bpm"],
                    "subgenero": guess["subgenero"],
                    "influencia": guess["influencia"],
                    "explicacao": guess["explicacao"],
                    "analise": tech_line,
                    "error": str(e),
                },
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        payload = {
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "influencia": "",
            "explicacao": "",
            "error": f"processing failed: {e.__class__.__name__}: {e}",
        }
        try:
            feats = locals().get("feats", None)
            if feats:
                payload["analise"] = build_tech_line(feats, chosen_src="fallback-error")
        except Exception:
            pass
        return JSONResponse(status_code=500, content=payload)
