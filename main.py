# -*- coding: utf-8 -*-
"""
saasdj-backend (Fase 1)

Aprimoramentos:
- Faixas FFT ajustadas: low=20–250, mid=250–4000, high=4000–20000
- onset_strength usa np.percentile(onset_env, 85)
- Campo 'confidence' removido da resposta
"""

from __future__ import annotations
import io, json, os
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

# Lista de subgêneros (universo permitido)
SUBGENRES: List[str] = [
    # House
    "Deep House","Tech House","Minimal Bass (Tech House)","Progressive House","Bass House",
    "Funky / Soulful House","Brazilian Bass","Future House","Afro House","Indie Dance",
    # Techno
    "Detroit Techno","Acid Techno","Industrial Techno","Peak Time Techno","Hard Techno",
    "Melodic Techno","High-Tech Minimal",
    # Trance
    "Uplifting Trance","Progressive Trance","Psytrance","Dark Psytrance",
    # EDM
    "Big Room","Progressive EDM",
    # Hard Dance
    "Hardstyle","Rawstyle","Gabber Hardcore","UK/Happy Hardcore","Jumpstyle",
    # Bass
    "Dubstep","Drum & Bass","Liquid DnB","Neurofunk"
]

# =============================================================================
# Funções de áudio
# =============================================================================
def load_audio(file_bytes: bytes, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Carrega 60 s do áudio, de 60–120 s (ou últimos 60 s se menor)."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_sec = len(audio) / 1000.0
    except Exception:
        audio, duration_sec = None, None

    if duration_sec is None:
        offset_seconds, duration_seconds = 60.0, 60.0
    elif duration_sec >= 120.0:
        offset_seconds, duration_seconds = 60.0, 60.0
    else:
        offset_seconds = max(0.0, duration_sec - 60.0)
        duration_seconds = min(60.0, duration_sec - offset_seconds)

    try:
        y, sr = librosa.load(
            io.BytesIO(file_bytes),
            sr=sr, mono=True,
            offset=offset_seconds, duration=duration_seconds
        )
        if y is None or len(y) == 0:
            raise ValueError("Áudio vazio após leitura principal")
        return y, sr
    except Exception:
        if audio is None:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        start_ms, end_ms = int(offset_seconds*1000), int((offset_seconds+duration_seconds)*1000)
        buf = io.BytesIO()
        audio[start_ms:end_ms].export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr

def extract_features(y: np.ndarray, sr: int) -> Dict[str, float | int | None]:
    """Extrai BPM, espectro e HP ratio."""
    try:
        from librosa.feature.rhythm import tempo as tempo_fn
        tempo_vals = tempo_fn(y=y, sr=sr, max_tempo=200, aggregate=None)
    except Exception:
        tempo_vals = librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=None)
    bpm = float(np.median(tempo_vals)) if tempo_vals is not None and len(tempo_vals)>0 else None
    if bpm and bpm < 90.0:
        bpm *= 2.0

    # FFT bandas ajustadas
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1 / sr)
    low = float(yf[(xf >= 20) & (xf < 250)].sum())
    mid = float(yf[(xf >= 250) & (xf < 4000)].sum())
    high = float(yf[(xf >= 4000) & (xf < 20000)].sum())
    total = max(low + mid + high, 1e-9)
    low_pct, mid_pct, high_pct = low/total, mid/total, high/total

    # HPSS e onset (percentil 85)
    H, P = librosa.effects.hpss(y)
    hp_ratio = (np.mean(np.abs(H))+1e-8)/(np.mean(np.abs(P))+1e-8)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = float(np.percentile(onset_env, 85)) if onset_env is not None and len(onset_env) else 0.0

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "low_pct": round(low_pct, 6),
        "mid_pct": round(mid_pct, 6),
        "high_pct": round(high_pct, 6),
        "hp_ratio": round(hp_ratio, 6),
        "onset_strength": round(onset_strength, 6),
    }

# =============================================================================
# Candidatos e GPT
# =============================================================================
def candidates_by_bpm(bpm: float | None) -> List[str]:
    margin = 4.0
    if bpm is None:
        return list(SUBGENRES)
    cands = []
    for name in SUBGENRES:
        rule = SOFT_RULES.get(name)
        if not rule:
            continue
        lo, hi = rule["bpm"]
        if (bpm >= lo - margin) and (bpm <= hi + margin):
            cands.append(name)
    return cands or list(SUBGENRES)

def call_gpt(features: Dict[str, float | int | None], candidates: List[str]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    user_payload = {"FEATURES": features, "CANDIDATES": candidates}
    messages = [
        {"role": "system", "content": "Você é um especialista em subgêneros de música eletrônica."},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=60)
    body = r.json()
    return body["choices"][0]["message"]["content"].strip()

# =============================================================================
# API FastAPI
# =============================================================================
app = FastAPI(title="saasdj-backend (Fase 1)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v1.1-fase1"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Recebe um .mp3/.wav, extrai features e retorna apenas BPM + subgênero."""
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        y, sr = load_audio(data)
        feats = extract_features(y, sr)
        bpm_val = feats.get("bpm")
        cands = candidates_by_bpm(bpm_val)

        try:
            content = call_gpt(feats, cands)
        except Exception as e:
            return JSONResponse(status_code=502, content={
                "bpm": int(round(bpm_val)) if bpm_val else None,
                "subgenero": "Subgênero Não Identificado",
                "error": str(e),
            })

        sub = "Subgênero Não Identificado"
        for line in content.splitlines():
            if "subgênero" in line.lower():
                sub = line.split(":", 1)[1].strip()
                break

        bpm_out = int(round(bpm_val)) if bpm_val is not None else None
        return {"bpm": bpm_out, "subgenero": sub}

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "error": f"processing failed: {e.__class__.__name__}: {e}",
        })
