# -*- coding: utf-8 -*-
"""
saasdj-backend (v1.2 – simples, assertivo e rápido)

Mudanças principais desta versão:
- Remove "confidence" do output. Agora retornamos: bpm (int), subgenero (str), explicacao (str), janela (str).
- Estratégia "Auto-Janela Única":
  * Fazemos um "skim" leve do áudio (até 6 minutos) para escolher automaticamente UMA janela de ~60s representativa
    (baseada em var(onset) + brilho/centroid). Evita diluir a assinatura do gênero.
  * Extraímos features COMPLETAS apenas da janela escolhida e mandamos para o GPT com um prompt mais aberto/estilístico.
- Features focadas e úteis ao GPT:
  * BPM (com dobra < 90)
  * spectral_centroid (médio), spectral_bandwidth (médio), zcr (médio), mfcc_mean[13]
  * low/mid/high_pct (só para contexto de espectro)
- Fallback simples (se o GPT falhar): filtra por BPM e escolhe por "caráter" (centroid/bandwidth/zcr), com desempate leve.

Requisitos (pip):
fastapi uvicorn librosa soundfile pydub scipy requests python-multipart
"""

from __future__ import annotations

import io
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import librosa
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# =============================================================================
# Config
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")
MAX_ANALYSIS_SECONDS = 360.0  # limite de 6 minutos para o "skim"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

# Universo permitido de subgêneros
SUBGENRES: List[str] = [
    # House
    "Deep House", "Tech House", "Minimal Bass (Tech House)", "Progressive House", "Bass House",
    "Funky / Soulful House", "Brazilian Bass", "Future House", "Afro House", "Indie Dance",
    # Techno
    "Detroit Techno", "Acid Techno", "Industrial Techno", "Peak Time Techno", "Hard Techno",
    "Melodic Techno", "High-Tech Minimal",
    # Trance
    "Uplifting Trance", "Progressive Trance", "Psytrance", "Dark Psytrance",
    # EDM (Festival)
    "Big Room", "Progressive EDM",
    # Hard Dance
    "Hardstyle", "Rawstyle", "Gabber Hardcore", "UK/Happy Hardcore", "Jumpstyle",
    # Bass Music
    "Dubstep", "Drum & Bass", "Liquid DnB", "Neurofunk",
]

# BPM ranges mínimos para filtrar candidatos no fallback (mantidos do projeto)
SOFT_BPM: Dict[str, Tuple[float, float]] = {
    "Deep House": (120,124), "Tech House": (124,128), "Minimal Bass (Tech House)": (124,128),
    "Progressive House": (122,128), "Bass House": (124,128), "Funky / Soulful House": (120,125),
    "Brazilian Bass": (120,126), "Future House": (124,128), "Afro House": (118,125), "Indie Dance": (110,125),

    "Detroit Techno": (122,130), "Acid Techno": (125,135), "Industrial Techno": (128,140),
    "Peak Time Techno": (128,132), "Hard Techno": (135,150), "Melodic Techno": (122,128),
    "High-Tech Minimal": (124,130),

    "Uplifting Trance": (134,140), "Progressive Trance": (132,138), "Psytrance": (138,146), "Dark Psytrance": (145,150),

    "Big Room": (126,128), "Progressive EDM": (124,128),

    "Hardstyle": (150,160), "Rawstyle": (150,160), "Gabber Hardcore": (170,190), "UK/Happy Hardcore": (165,180),
    "Jumpstyle": (140,150),

    "Dubstep": (136,146), "Drum & Bass": (170,180), "Liquid DnB": (170,176), "Neurofunk": (172,178),
}

# Hints qualitativos para fallback (caráter por brilho/abertura/vocais via ZCR)
STYLE_HINTS: Dict[str, Dict[str, str]] = {
    # valores esperados: 'low' | 'mid' | 'high' (ou 'mid_high', 'low_mid' em alguns)
    "Deep House": {"brightness":"mid","bandwidth":"mid","vocals":"mid"},
    "Tech House": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Minimal Bass (Tech House)": {"brightness":"low","bandwidth":"low","vocals":"low"},
    "Progressive House": {"brightness":"mid_high","bandwidth":"high","vocals":"mid"},
    "Bass House": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Funky / Soulful House": {"brightness":"mid","bandwidth":"mid","vocals":"high"},
    "Brazilian Bass": {"brightness":"low_mid","bandwidth":"low_mid","vocals":"mid"},
    "Future House": {"brightness":"high","bandwidth":"high","vocals":"mid"},
    "Afro House": {"brightness":"mid","bandwidth":"mid","vocals":"mid_high"},
    "Indie Dance": {"brightness":"mid","bandwidth":"mid_high","vocals":"mid"},

    "Detroit Techno": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Acid Techno": {"brightness":"mid_high","bandwidth":"mid_high","vocals":"low"},
    "Industrial Techno": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Peak Time Techno": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Hard Techno": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Melodic Techno": {"brightness":"high","bandwidth":"high","vocals":"mid"},
    "High-Tech Minimal": {"brightness":"mid","bandwidth":"mid","vocals":"low"},

    "Uplifting Trance": {"brightness":"high","bandwidth":"high","vocals":"mid_high"},
    "Progressive Trance": {"brightness":"mid_high","bandwidth":"high","vocals":"mid"},
    "Psytrance": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Dark Psytrance": {"brightness":"low_mid","bandwidth":"mid","vocals":"low"},

    "Big Room": {"brightness":"high","bandwidth":"high","vocals":"mid"},
    "Progressive EDM": {"brightness":"high","bandwidth":"high","vocals":"mid_high"},

    "Hardstyle": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Rawstyle": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Gabber Hardcore": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "UK/Happy Hardcore": {"brightness":"high","bandwidth":"high","vocals":"high"},
    "Jumpstyle": {"brightness":"mid_high","bandwidth":"mid","vocals":"low"},

    "Dubstep": {"brightness":"mid","bandwidth":"high","vocals":"low"},
    "Drum & Bass": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Liquid DnB": {"brightness":"mid_high","bandwidth":"high","vocals":"mid_high"},
    "Neurofunk": {"brightness":"high","bandwidth":"high","vocals":"low"},
}


# =============================================================================
# Utils de áudio
# =============================================================================

def _get_duration(file_bytes: bytes) -> Optional[float]:
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        return len(audio) / 1000.0
    except Exception:
        return None

def _tempo_detect(y: np.ndarray, sr: int) -> Optional[float]:
    try:
        from librosa.feature.rhythm import tempo as tempo_fn
        vals = tempo_fn(y=y, sr=sr, max_tempo=200, aggregate=None)
    except Exception:
        vals = librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=None)
    if vals is None or len(vals) == 0:
        return None
    bpm = float(np.median(vals))
    if bpm < 90.0:
        bpm *= 2.0
    return bpm

def _fft_bands(y: np.ndarray, sr: int) -> Tuple[float,float,float,float,float,float]:
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1/sr)
    low = float(yf[(xf >= 20) & (xf < 120)].sum())
    mid = float(yf[(xf >= 120) & (xf < 2000)].sum())
    high = float(yf[(xf >= 2000)].sum())
    total = max(low + mid + high, 1e-9)
    return low, mid, high, float(low/total), float(mid/total), float(high/total)

def _zcr_mean(y: np.ndarray) -> float:
    z = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(z)) if z is not None else 0.0

def _mfcc_mean(y: np.ndarray, sr: int, n_mfcc: int = 13) -> List[float]:
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return [float(np.mean(mf[i])) for i in range(n_mfcc)]

def _load_any(file_bytes: bytes, offset: float, duration: float, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Carrega mono+resample com offset/duration; fallback via pydub."""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True, offset=offset, duration=duration)
        if y is None or len(y) == 0:
            raise ValueError("vazio")
        return y, sr
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg = audio[int(offset*1000): int((offset+duration)*1000)]
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr

# =============================================================================
# Seleção de UMA janela: skim leve + score
# =============================================================================

def _skim_signals(file_bytes: bytes, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Carrega até 6 minutos e calcula:
      - onset strength por frame
      - spectral centroid por frame
    Retorna: (onset_env, centroid, duration_sec, sr)
    """
    duration = _get_duration(file_bytes)
    target = min(MAX_ANALYSIS_SECONDS, duration if duration else MAX_ANALYSIS_SECONDS)

    # carrega o "skim" (até 6 min), sem mono stacking extra (librosa já mono=True)
    y, sr = _load_any(file_bytes, offset=0.0, duration=target, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc = sc[0] if sc is not None and sc.shape[0] > 0 else np.zeros_like(onset_env)

    return onset_env, sc, float(len(y)/sr), sr

def _frame_times(n_frames: int, hop_length: int, sr: int) -> np.ndarray:
    return (np.arange(n_frames) * hop_length) / sr

def _select_best_window(file_bytes: bytes, win_sec: float = 60.0, step_sec: float = 5.0, sr: int = 22050) -> Tuple[float, float, str]:
    """
    Seleciona a melhor janela [start, start+win_sec] baseada em score:
      score = 0.6 * var(onset) + 0.4 * mean(centroid_norm)
    Normalizamos centroid em [0,1] por percentis globais para robustez.
    Retorna: (start_sec, duration_sec_real, label_intro_meio_final)
    """
    onset_env, sc, dur, sr = _skim_signals(file_bytes, sr=sr)

    # mapear frames para tempo
    hop_length = 512  # default interno do librosa.onset_strength
    # onset_strength usa hop_length=512 por padrão; confirmamos para cálculo de janelas
    t = _frame_times(len(onset_env), hop_length, sr)

    # normalização robusta do centroid
    p1, p99 = np.percentile(sc, 1), np.percentile(sc, 99)
    sc_norm = np.clip((sc - p1) / max(p99 - p1, 1e-6), 0, 1)

    # var(onset) por janela + mean(centroid_norm) por janela
    starts = np.arange(0.0, max(0.0, dur - win_sec) + 1e-9, step_sec)
    best_score, best_start = -1.0, 0.0

    for s in starts:
        e = min(dur, s + win_sec)
        # frames na janela [s, e]
        idx = np.where((t >= s) & (t <= e))[0]
        if len(idx) < 8:
            continue
        var_onset = float(np.var(onset_env[idx]))
        mean_sc = float(np.mean(sc_norm[idx]))
        score = 0.6 * var_onset + 0.4 * mean_sc
        if score > best_score:
            best_score, best_start = score, s

    # se áudio menor que win_sec, usa tudo
    if dur <= win_sec + 1.0 or best_score < 0:
        start = 0.0
        duration = min(win_sec, dur)
    else:
        start = best_start
        duration = min(win_sec, max(0.0, dur - start))

    # rótulo humano da janela
    label = "meio"
    if start <= 30.0:
        label = "intro"
    elif (dur - (start + duration)) <= 30.0:
        label = "final"

    return float(start), float(duration), label

# =============================================================================
# Extração de features COMPLETAS (apenas da janela escolhida)
# =============================================================================

def extract_features_window(file_bytes: bytes, start: float, duration: float, sr: int = 22050) -> Dict[str, float | int | List[float] | None]:
    y, sr = _load_any(file_bytes, offset=start, duration=duration, sr=sr)

    # BPM
    bpm = _tempo_detect(y, sr)

    # Espectro (pcts)
    low, mid, high, lp, mp, hp = _fft_bands(y, sr)

    # TimbraIS
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = _zcr_mean(y)
    mfcc_vec = _mfcc_mean(y, sr, n_mfcc=13)

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "spectral_centroid": round(centroid, 3),
        "spectral_bandwidth": round(bandwidth, 3),
        "zcr": round(zcr, 6),
        "mfcc_mean": [round(v, 6) for v in mfcc_vec],
        "low_pct": round(lp, 6), "mid_pct": round(mp, 6), "high_pct": round(hp, 6),
        "window": {"start": start, "duration": duration}
    }

# =============================================================================
# Heurísticas simples (fallback)
# =============================================================================

def _class_from_value(v: Optional[float], low_thr: float, high_thr: float, midband: Optional[Tuple[float,float]] = None) -> str:
    if v is None:
        return "mid"
    if v < low_thr:
        return "low"
    if v > high_thr:
        return "high"
    if midband:
        lo, hi = midband
        if v > (hi - (hi-lo)*0.25): return "mid_high"
        if v < (lo + (hi-lo)*0.25): return "low_mid"
    return "mid"

def _bpm_candidates(bpm: Optional[float]) -> List[str]:
    if bpm is None:
        return SUBGENRES[:]
    margin = 4.0
    cands = []
    for name, (lo, hi) in SOFT_BPM.items():
        if (bpm >= lo - margin) and (bpm <= hi + margin):
            cands.append(name)
    return cands or SUBGENRES[:]

def fallback_simple(feats: Dict[str, float | int | List[float] | None]) -> str:
    """
    Fallback muito simples:
      1) Filtra por BPM.
      2) Matching qualitativo de centroid/bandwidth/zcr com STYLE_HINTS.
      3) Empate → primeiro por faixa exata de BPM (sem margem), senão mantém ordem.
    """
    bpm = feats.get("bpm")
    centroid = feats.get("spectral_centroid")
    bandwidth = feats.get("spectral_bandwidth")
    zcr = feats.get("zcr")

    c_class = _class_from_value(centroid, low_thr=1200.0, high_thr=1800.0, midband=(1200.0,1800.0))
    b_class = _class_from_value(bandwidth, low_thr=1400.0, high_thr=2000.0, midband=(1400.0,2000.0))
    v_class = _class_from_value(zcr, low_thr=0.05, high_thr=0.10)

    cands = _bpm_candidates(bpm)

    def _match_score(name: str) -> int:
        hint = STYLE_HINTS.get(name, {})
        s = 0
        if hint.get("brightness") == c_class: s += 1
        if hint.get("bandwidth") == b_class: s += 1
        if hint.get("vocals") == v_class: s += 1
        return s

    cands_sorted = sorted(cands, key=lambda n: _match_score(n), reverse=True)

    # Tenta reforçar quem está estritamente dentro da faixa BPM sem margem
    if bpm is not None:
        strict = []
        for n in cands_sorted:
            lo, hi = SOFT_BPM.get(n, (0, 999))
            if lo <= bpm <= hi:
                strict.append(n)
        if strict:
            cands_sorted = strict

    return cands_sorted[0] if cands_sorted else "Subgênero Não Identificado"


# =============================================================================
# LLM (prompt aberto + explicação)
# =============================================================================

PROMPT = """
Você é um especialista em música eletrônica.
Receberá features de UMA JANELA representativa (≈60s) da faixa.
Classifique em exatamente UM subgênero da lista fornecida ou retorne 'Subgênero Não Identificado' se não couber.

Use interpretação estilística:
- BPM como filtro grosso (ex.: ~128 = House, 130+ = Techno, 140+ = Trance, 150+ = Hard Dance, ~140 half-time = Dubstep).
- spectral_centroid (brilho) e spectral_bandwidth (abertura) para diferenciar sons secos/minimalistas de sons atmosféricos/brilhantes.
- zcr como proxy de presença de vocais/ruído.
- mfcc_mean[13] como assinatura timbral (sem calcular distância; só para complementar sua intuição estilística).
- low/mid/high_pct apenas como contexto de distribuição de energia.

Responda EXACTAMENTE em DUAS linhas:
Subgênero: <um da lista de CANDIDATES ou 'Subgênero Não Identificado'>
Explicação: <1–3 frases musicais, sem números mágicos, justificando a escolha>

Notas internas (não exponha):
- Absorções: Bass House ← Electro House; Uplifting/Progressive Trance ← Vocal Trance; Psytrance ← Goa; Dubstep ← Riddim.
- Prefira não chutar: se não houver encaixe claro, use 'Subgênero Não Identificado'.
"""

def call_gpt_open(features: Dict[str, float | int | List[float] | None], candidates: List[str]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "FEATURES": {
            "bpm": features.get("bpm"),
            "spectral_centroid": features.get("spectral_centroid"),
            "spectral_bandwidth": features.get("spectral_bandwidth"),
            "zcr": features.get("zcr"),
            "mfcc_mean": features.get("mfcc_mean"),
            "low_pct": features.get("low_pct"),
            "mid_pct": features.get("mid_pct"),
            "high_pct": features.get("high_pct"),
        },
        "CANDIDATES": candidates,
    }
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
    return body["choices"][0]["message"]["content"].strip()


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="saasdj-backend (v1.2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v1.2"}

@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    mode: str = Query("auto", description="auto | intro | meio | final (força janela)"),
):
    """
    Recebe .mp3/.wav, escolhe UMA janela representativa (~60s) e classifica via GPT (prompt aberto).
    Retorno: {"bpm": <int|None>, "subgenero": <str>, "explicacao": <str>, "janela": <'intro'|'meio'|'final'>}
    """
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # Seleção de janela
        duration = _get_duration(data) or MAX_ANALYSIS_SECONDS
        if mode == "auto":
            start, dur, label = _select_best_window(data, win_sec=60.0, step_sec=5.0, sr=22050)
        else:
            # Força janelas clássicas para debug
            if mode == "intro":
                start, dur, label = 0.0, min(60.0, duration), "intro"
            elif mode == "final":
                start = max(0.0, duration - 60.0)
                dur = duration - start
                label = "final"
            else:  # "meio" default
                if duration >= 120.0:
                    start, dur, label = 60.0, 60.0, "meio"
                else:
                    mid = max(0.0, (duration/2.0) - 30.0)
                    start, dur, label = mid, min(60.0, duration - mid), "meio"

        # Features completas da janela escolhida
        feats = extract_features_window(data, start=start, duration=dur, sr=22050)

        # Candidatos por BPM (filtro grosso)
        bpm_val = feats.get("bpm")
        candidates = _bpm_candidates(bpm_val)

        # Chamar GPT
        try:
            content = call_gpt_open(feats, candidates)
        except Exception as e:
            # Fallback simples
            fb_sub = fallback_simple(feats)
            bpm_out = int(round(bpm_val)) if bpm_val is not None else None
            return JSONResponse(status_code=502, content={
                "bpm": bpm_out,
                "subgenero": fb_sub if fb_sub in SUBGENRES else "Subgênero Não Identificado",
                "explicacao": "Classificação feita por heurística local (fallback) devido a erro no LLM.",
                "janela": label,
                "error": str(e),
            })

        # Parse: "Subgênero: ..." / "Explicação: ..."
        sub = "Subgênero Não Identificado"
        explic = ""
        for line in content.splitlines():
            L = line.strip()
            Lc = L.lower()
            if Lc.startswith("subgênero:") or Lc.startswith("subgenero:") or Lc.startswith("subgénero:"):
                sub = L.split(":", 1)[1].strip()
            elif Lc.startswith("explicação:") or Lc.startswith("explicacao:"):
                explic = L.split(":", 1)[1].strip()

        # Sanitização
        if sub != "Subgênero Não Identificado" and sub not in SUBGENRES:
            sub = "Subgênero Não Identificado"

        bpm_out = int(round(bpm_val)) if bpm_val is not None else None
        return {"bpm": bpm_out, "subgenero": sub, "explicacao": explic, "janela": label}

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "explicacao": "Falha inesperada no processamento.",
            "janela": "meio",
            "error": f"processing failed: {e.__class__.__name__}: {e}",
        })
