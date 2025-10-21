# -*- coding: utf-8 -*-
"""
saasdj-backend (organizado v3)

- FastAPI para classificar subgênero de música eletrônica a partir de .mp3/.wav
- Extrai features com librosa (BPM robusto focado no percussivo, bandas espectrais corrigidas,
  HP ratio, onset normalizado, kickband)
- Usa regras numéricas + GPT para eleger o subgênero
- Resposta SEMPRE inclui: {"bpm": <int|None>, "subgenero": <str>, "analise": "<linha técnica>"}

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

# Lista oficial de subgêneros (universo permitido)
SUBGENRES: List[str] = [
    # House
    "Deep House",
    "Tech House",
    "Minimal Bass (Tech House)",
    "Progressive House",
    "Bass House",
    "Funky / Soulful House",
    "Brazilian Bass",
    "Future House",
    "Afro House",
    "Indie Dance",
    # Techno
    "Detroit Techno",
    "Acid Techno",
    "Industrial Techno",
    "Peak Time Techno",
    "Hard Techno",
    "Melodic Techno",
    "High-Tech Minimal",
    # Trance
    "Uplifting Trance",
    "Progressive Trance",
    "Psytrance",
    "Dark Psytrance",
    # EDM (Festival)
    "Big Room",
    "Progressive EDM",
    # Hard Dance
    "Hardstyle",
    "Rawstyle",
    "Gabber Hardcore",
    "UK/Happy Hardcore",
    "Jumpstyle",
    # Bass Music
    "Dubstep",
    "Drum & Bass",
    "Liquid DnB",
    "Neurofunk",
]

# ---- SOFT RULES NUMÉRICAS (por subgênero) ----
# Bandas (proporções 0–1): low=20–250 Hz | mid=250–4000 Hz | high=4000–20000 Hz
# hp_ratio = H/P (harmônico ÷ percussivo); onset_strength = média normalizada do onset
SOFT_RULES: Dict[str, Dict] = {
    # ---------------- HOUSE ----------------
    "Deep House": {
        "bpm": (120, 124),
        "bands_pct": {"low": (0.20, 0.35), "mid": (0.40, 0.60), "high": (0.10, 0.25)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.20, 0.50),
        "signatures": "groove suave/profundo, acordes/pads, vocais quentes; menos foco em transientes",
    },
    "Tech House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.12, 0.36)},  # high max 0.35
        "hp_ratio": (0.75, 1.25),  # era até 1.05
        "onset_strength": (0.40, 0.65),
        "signatures": "kick/bass secos e funcionais, grooves repetitivos, poucos leads",
    },
    "Minimal Bass (Tech House)": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.45, 0.68), "mid": (0.16, 0.32), "high": (0.08, 0.22)},
        "hp_ratio": (0.70, 0.95),
        "onset_strength": (0.35, 0.60),
        "signatures": "sub muito forte e arranjo minimalista; foco no baixo e groove enxuto",
    },
    "Progressive House": {
        "bpm": (122, 128),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.58), "high": (0.15, 0.28)},  # high max 0.28
        "hp_ratio": (1.10, 1.70),
        "onset_strength": (0.25, 0.50),  # mantém teto 0.50
        "signatures": "builds longos, atmosfera melódica, progressão constante/emotiva",
    },
    "Bass House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.45, 0.70), "mid": (0.20, 0.40), "high": (0.18, 0.35)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.45, 0.70),
        "signatures": "basslines agressivas/‘talking’, queda forte no drop (absorve Electro House)",
    },
    "Funky / Soulful House": {
        "bpm": (120, 125),
        "bands_pct": {"low": (0.25, 0.40), "mid": (0.40, 0.60), "high": (0.10, 0.25)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.20, 0.45),
        "signatures": "instrumentação orgânica, elementos soul/disco, presença de vocais",
    },
    "Brazilian Bass": {
        "bpm": (120, 126),
        "bands_pct": {"low": (0.50, 0.75), "mid": (0.18, 0.35), "high": (0.08, 0.22)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.35, 0.60),
        "signatures": "sub/slap marcante com groove pop-friendly, vocais ocasionais",
    },
    "Future House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.35, 0.55), "mid": (0.22, 0.40), "high": (0.20, 0.38)},
        "hp_ratio": (0.95, 1.25),
        "onset_strength": (0.45, 0.70),
        "signatures": "timbres ‘future’/serrilhas, drops claros e brilhantes",
    },
    "Afro House": {
        "bpm": (118, 125),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.35, 0.55), "high": (0.12, 0.28)},
        "hp_ratio": (1.05, 2.20),  # era até 1.60
        "onset_strength": (0.35, 0.60),
        "signatures": "percussões afro, groove orgânico, vocais/texturas étnicas",
    },
    "Indie Dance": {
        "bpm": (110, 125),
        "bands_pct": {"low": (0.18, 0.35), "mid": (0.40, 0.60), "high": (0.12, 0.32)},  # high max 0.32
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.25, 0.50),
        "signatures": "vibe retrô/alternativa, synths vintage, menos ênfase em transientes",
    },
    # ---------------- TECHNO ----------------
    "Detroit Techno": {
        "bpm": (122, 130),
        "bands_pct": {"low": (0.28, 0.45), "mid": (0.30, 0.50), "high": (0.12, 0.28)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.40, 0.65),
        "signatures": "groove clássico/analógico, estética quente, linhas repetitivas",
    },
    "Acid Techno": {
        "bpm": (125, 135),
        "bands_pct": {"low": (0.28, 0.45), "mid": (0.35, 0.55), "high": (0.15, 0.32)},
        "hp_ratio": (0.95, 1.30),
        "onset_strength": (0.45, 0.70),
        "signatures": "timbre TB-303 em destaque (ressonante/squelch) como elemento central",
    },
    "Industrial Techno": {
        "bpm": (128, 140),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.20, 0.40)},
        "hp_ratio": (0.75, 1.05),
        "onset_strength": (0.60, 0.85),
        "signatures": "texturas industriais/ruidosas, sensação ‘fábrica’, percussão pesada",
    },
    "Peak Time Techno": {
        "bpm": (128, 136),  # era até 132
        "bands_pct": {"low": (0.32, 0.56), "mid": (0.24, 0.42), "high": (0.18, 0.35)},
        "hp_ratio": (0.85, 1.15),
        "onset_strength": (0.55, 0.80),
        "signatures": "4x4 direto para ápice; leads discretos; energia constante (ref. Victor Ruiz – All Night Long)",
    },
    "Hard Techno": {
        "bpm": (135, 165),  # era até 150
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.70, 1.30),  # era até 0.95
        "onset_strength": (0.65, 0.90),
        "signatures": "agressivo e percussivo; kicks duros; pouca melodia",
    },
    "Melodic Techno": {
        "bpm": (122, 128),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.60), "high": (0.15, 0.32)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.30, 0.60),
        "signatures": "pads/leads emocionais e cinematográficos, progressão envolvente",
    },
    "High-Tech Minimal": {
        "bpm": (124, 130),
        "bands_pct": {"low": (0.30, 0.58), "mid": (0.22, 0.40), "high": (0.10, 0.25)},  # low mais amplo
        "hp_ratio": (0.85, 1.60),  # era até 1.15
        "onset_strength": (0.35, 0.60),
        "signatures": "minimalista, design sonoro detalhista; sub consistente, timbres enxutos",
    },
    # ---------------- TRANCE ----------------
    "Uplifting Trance": {
        "bpm": (134, 140),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.58), "high": (0.22, 0.38)},  # high mínimo 0.22 (↑)
        "hp_ratio": (1.30, 2.20),  # ligeiro ↑
        "onset_strength": (0.60, 0.90),  # ↑ mais impacto
        "signatures": "...",
    },
    "Progressive Trance": {
        "bpm": (132, 138),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.40, 0.62), "high": (0.15, 0.26)},  # high máximo 0.26 (↓)
        "hp_ratio": (1.10, 1.90),
        "onset_strength": (0.25, 0.60),  # teto menor que Uplifting
        "signatures": "...",
    },
    "Psytrance": {
        "bpm": (138, 146),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.32, 0.52), "high": (0.15, 0.30)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.35, 0.60),
        "signatures": "rolling bass e FX psicodélicos; absorve Goa",
    },
    "Dark Psytrance": {
        "bpm": (145, 150),
        "bands_pct": {"low": (0.30, 0.50), "mid": (0.25, 0.45), "high": (0.20, 0.40)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.50, 0.75),
        "signatures": "mais escuro/denso; texturas e ruídos em destaque",
    },
    # ---------------- EDM (FESTIVAL) ----------------
    "Big Room": {
        "bpm": (126, 128),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.60, 0.85),
        "signatures": "drops marcados; dinâmica alta; hi-hats/brilho dando impacto",
    },
    "Progressive EDM": {
        "bpm": (126, 130),  # era 124–128
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.40, 0.60), "high": (0.25, 0.40)},  # high maior
        "hp_ratio": (1.40, 2.20),  # mais melódico/brilhante
        "onset_strength": (0.35, 0.60),
        "signatures": "melódico de festival, polido, progressivo/pop-friendly",
    },
    # ---------------- HARD DANCE ----------------
    "Hardstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.22, 0.42)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.65, 0.90),
        "signatures": "kicks distorcidos/‘reverse bass’, intensidade alta",
    },
    "Rawstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.25, 0.45)},
        "hp_ratio": (0.75, 1.00),
        "onset_strength": (0.70, 0.95),
        "signatures": "mais agressivo/distorcido que Hardstyle; sound design extremo",
    },
    "Gabber Hardcore": {
        "bpm": (170, 190),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.20, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.70, 0.95),
        "onset_strength": (0.75, 0.98),
        "signatures": "muito agressivo, kicks ‘serra’, textura crua",
    },
    "UK/Happy Hardcore": {
        "bpm": (165, 180),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.35, 0.55), "high": (0.22, 0.42)},
        "hp_ratio": (1.00, 1.60),
        "onset_strength": (0.60, 0.85),
        "signatures": "rápido, eufórico/brilhante, melodias felizes",
    },
    "Jumpstyle": {
        "bpm": (140, 150),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.25, 0.45), "high": (0.15, 0.32)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.55, 0.80),
        "signatures": "padrões rítmicos ‘saltados’, batidas quadradas",
    },
    # ---------------- BASS MUSIC ----------------
    "Dubstep": {
        "bpm": (136, 146),
        "bands_pct": {"low": (0.45, 0.75), "mid": (0.20, 0.40), "high": (0.20, 0.40)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.60, 0.85),
        "signatures": "half-time ~140, wobble/sub pesado; absorve Riddim",
    },
    "Drum & Bass": {
        "bpm": (170, 180),
        "bands_pct": {"low": (0.30, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.75, 0.98),
        "signatures": "breakbeat rápido, muitos ataques, energia constante",
    },
    "Liquid DnB": {
        "bpm": (170, 176),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.38, 0.58), "high": (0.15, 0.30)},
        "hp_ratio": (1.05, 1.60),
        "onset_strength": (0.45, 0.70),
        "signatures": "suave/atmosférico, pads/vocais; break mais limpo",
    },
    "Neurofunk": {
        "bpm": (172, 178),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.65, 0.90),
        "signatures": "baixo serrado/complexo, agressivo e técnico",
    },
}


# =============================================================================
# Carregamento multi-janela
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


def load_audio_windows(file_bytes: bytes, sr: int = 22050) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Retorna 3 janelas:
    - mid60:   60–120s (ou últimos 60s se a faixa < 120s)
    - center30: 30s centrados na faixa
    - last60:  últimos 60s (ou desde 0 se <60s)
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_sec = len(audio) / 1000.0
    except Exception:
        audio = None
        duration_sec = None

    windows: Dict[str, Tuple[np.ndarray, int]] = {}

    # mid60
    if duration_sec is None or duration_sec >= 120.0:
        off_mid60, dur_mid60 = 60.0, 60.0
    else:
        off_mid60 = max(0.0, duration_sec - 60.0)
        dur_mid60 = min(60.0, duration_sec - off_mid60)
    windows["mid60"] = _load_segment(file_bytes, off_mid60, dur_mid60, sr)

    # center30
    if duration_sec is None:
        # fallback: recorta do mid60
        y_mid, sr_mid = windows["mid60"]
        n = len(y_mid)
        start = max(0, n // 2 - int(15 * sr_mid))
        end = min(n, start + int(30 * sr_mid))
        windows["center30"] = (y_mid[start:end], sr_mid)
    else:
        start_center = max(0.0, (duration_sec / 2.0) - 15.0)
        windows["center30"] = _load_segment(file_bytes, start_center, min(30.0, duration_sec), sr)

    # last60
    if duration_sec is None or duration_sec <= 60.0:
        off_last, dur_last = 0.0, min(60.0, duration_sec or 60.0)
    else:
        off_last, dur_last = duration_sec - 60.0, 60.0
    windows["last60"] = _load_segment(file_bytes, off_last, dur_last, sr)

    return windows


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
    kick = float(yf[(xf >= 40) & (xf < 100)].sum())  # ajuda Tech House/PeakTime/Minimal Bass

    return low, mid, high, kick


def _tempo_candidates(y: np.ndarray, sr: int) -> List[float]:
    """Estimativas de BPM focadas no componente percussivo (reduz viés melódico)."""
    bpms: List[float] = []
    try:
        H, P = librosa.effects.hpss(y)
    except Exception:
        H, P = None, None

    # 1) beat_track no percussivo (prioritário)
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


def _choose_bpm_from_rules(bpm_cands: List[float], rules: Dict[str, Dict]) -> float | None:
    """Escolhe o BPM que melhor se encaixa nas faixas dos subgêneros (soma de aderências)."""
    if not bpm_cands:
        return None

    def score_bpm(b: float, rng: Tuple[float, float]) -> float:
        lo, hi = rng
        if b < lo:
            return max(0.0, 1.0 - (lo - b) / (hi - lo + 1e-6))
        if b > hi:
            return max(0.0, 1.0 - (b - hi) / (hi - lo + 1e-6))
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 + 1e-6
        return 1.0 + 0.2 * (1.0 - abs(b - mid) / half)

    best_bpm, best_sum = None, -1.0
    for b in bpm_cands:
        s = 0.0
        for meta in rules.values():
            s += score_bpm(b, meta["bpm"])
        if s > best_sum:
            best_sum = s
            best_bpm = b
    return best_bpm


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float | int | None]:
    """Features de UMA janela."""
    # BPM robusto
    bpm_candidates = _tempo_candidates(y, sr)
    bpm = _choose_bpm_from_rules(bpm_candidates, SOFT_RULES)

    # Espectro
    low, mid, high, kick = _bands_energy(y, sr)
    total = max(low + mid + high, 1e-9)
    low_pct = float(low / total)
    mid_pct = float(mid / total)
    high_pct = float(high / total)

    # HPSS e onset normalizado
    H, P = librosa.effects.hpss(y)
    hp_ratio = float((np.mean(np.abs(H)) + 1e-8) / (np.mean(np.abs(P)) + 1e-8))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env is not None and len(onset_env):
        onset_strength = float(np.mean(onset_env) / (np.std(onset_env) + 1e-6))
    else:
        onset_strength = 0.0

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


def extract_features_multi(windows: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, float | int | None]:
    """
    Combina 3 janelas com PESOS:
    mid60=0.2, center30=0.3, last60=0.5 (mais peso ao final).
    BPM = mediana ponderada aproximada (via repetição por peso).
    Demais = média ponderada.
    """
    order = [("mid60", 0.15), ("center30", 0.25), ("last60", 0.60)]
    feats = []
    for key, _w in order:
        y, sr = windows[key]
        feats.append(extract_features(y, sr))

    # BPM: "mediana ponderada" simples (replicando valores por peso*10)
    bpm_values = []
    for (key, w), f in zip(order, feats):
        if f.get("bpm") is not None:
            bpm_values += [f["bpm"]] * max(1, int(round(w * 10)))
    bpm = float(np.median(bpm_values)) if bpm_values else None

    def wavg(k):
        vals, ws = [], []
        for (key, w), f in zip(order, feats):
            v = f.get(k)
            if v is not None:
                vals.append(v); ws.append(w)
        return float(np.average(vals, weights=ws)) if vals else None

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "energy_low": wavg("energy_low"),
        "energy_mid": wavg("energy_mid"),
        "energy_high": wavg("energy_high"),
        "kick_40_100": wavg("kick_40_100"),
        "low_pct": wavg("low_pct"),
        "mid_pct": wavg("mid_pct"),
        "high_pct": wavg("high_pct"),
        "hp_ratio": wavg("hp_ratio"),
        "onset_strength": wavg("onset_strength"),
    }


# === Helper: linha técnica na resposta ===
def _fmt_pct(x: float | None) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"

def _fmt_float(x: float | None, nd=2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"

def build_tech_line(
    feats: Dict[str, float | int | None],
    cands: List[str],
    chosen: str,
    decision_source: str,
) -> str:
    """
    Retorna UMA linha com os principais dados extraídos e contexto da decisão.
    Ex.: BPM=128; low%=35.2%; mid%=44.1%; high%=20.7%; hp=1.12; onset=0.53; kick=1234567; cands=[Tech House,...]; chosen=Tech House; source=llm
    """
    bpm = feats.get("bpm")
    low_pct = feats.get("low_pct")
    mid_pct = feats.get("mid_pct")
    high_pct = feats.get("high_pct")
    hp = feats.get("hp_ratio")
    onset = feats.get("onset_strength")
    kick = feats.get("kick_40_100")

    parts = [
        f"BPM={int(round(bpm)) if bpm is not None else 'n/a'}",
        f"low%={_fmt_pct(low_pct)}",
        f"mid%={_fmt_pct(mid_pct)}",
        f"high%={_fmt_pct(high_pct)}",
        f"hp={_fmt_float(hp,2)}",
        f"onset={_fmt_float(onset,2)}",
        f"kick={int(kick) if kick is not None else 'n/a'}",
        f"cands=[{', '.join(cands)}]",
        f"chosen={chosen}",
        f"source={decision_source}",
    ]
    return "; ".join(parts)


# =============================================================================
# Seleção de candidatos e fallback (backend)
# =============================================================================

def candidates_by_bpm(bpm: float | None) -> List[str]:
    """Seleciona candidatos pela faixa de BPM com margens mais duras (2→3)."""
    if bpm is None:
        return list(SOFT_RULES.keys())

    def in_margin(b, rng, m):
        lo, hi = rng
        return (b >= lo - m) and (b <= hi + m)

    for margin in (2.0, 3.0):
        cands = [name for name, meta in SOFT_RULES.items() if in_margin(bpm, meta["bpm"], margin)]
        if cands:
            return cands
    return list(SOFT_RULES.keys())


def backend_fallback_best_candidate(features: Dict[str, float | int | None], candidates: List[str]) -> str:
    """Se o LLM falhar, escolhe o melhor candidato (sem confidence)."""
    bpm = features.get("bpm")
    lp = features.get("low_pct", 0.0) or 0.0
    mp = features.get("mid_pct", 0.0) or 0.0
    hpv = features.get("high_pct", 0.0) or 0.0
    hpr = features.get("hp_ratio", 0.0) or 0.0
    kick = features.get("kick_40_100", 0.0) or 0.0

    def _score_in_range(val: float | None, rng: Tuple[float, float]) -> float:
        lo, hi = rng
        if val is None:
            return 0.0
        if val < lo:
            return max(0.0, 1.0 - (lo - val) / (hi - lo + 1e-6))
        if val > hi:
            return max(0.0, 1.0 - (val - hi) / (hi - lo + 1e-6))
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 + 1e-6
        return 1.0 + max(0.0, 0.2 * (1.0 - abs(val - mid) / half))

    best_name = "Subgênero Não Identificado"
    best_score = 0.0
    for name in candidates:
        rule = SOFT_RULES[name]
        s_bpm = _score_in_range(bpm, rule["bpm"])
        s_low = _score_in_range(lp, rule["bands_pct"]["low"])
        s_mid = _score_in_range(mp, rule["bands_pct"]["mid"])
        s_high = _score_in_range(hpv, rule["bands_pct"]["high"])
        s_hp = _score_in_range(hpr, rule["hp_ratio"])

        bands_avg = (s_low + s_mid + s_high) / 3.0
        score = 0.45 * s_bpm + 0.40 * bands_avg + 0.15 * s_hp

        # bônus suave se o kick está forte (gêneros 4x4 voltados à pista)
        if name in ("Tech House", "Peak Time Techno", "Minimal Bass (Tech House)", "Hard Techno", "High-Tech Minimal") and kick > 0:
            score *= 1.03  # leve viés pró pista
      
      # desempate pró Melodic Techno quando bem melódica
        if name == "Melodic Techno":
            if (hpr >= 1.60) and (mp >= 0.45) and (120 <= (bpm or 0) <= 128):
                score *= 1.04  # viés leve pró MT em casos "claros"
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < 0.40:
        return "Subgênero Não Identificado"
    return best_name


# =============================================================================
# Formatação para o LLM
# =============================================================================

def format_rules_for_candidates(cands: List[str]) -> str:
    """Constrói um texto compacto com faixas numéricas para os candidatos."""
    lines = []
    for name in cands:
        m = SOFT_RULES[name]
        lo_bpm, hi_bpm = m["bpm"]
        lp = m["bands_pct"]["low"]
        mp = m["bands_pct"]["mid"]
        hp = m["bands_pct"]["high"]
        hr_lo, hr_hi = m["hp_ratio"]
        lines.append(
            "Nome: {name}\n"
            "BPM: {lo}-{hi}\n"
            "Bandas%: low={lp0:.2f}-{lp1:.2f} | mid={mp0:.2f}-{mp1:.2f} | high={hp0:.2f}-{hp1:.2f}\n"
            "HP Ratio: {hr0:.2f}-{hr1:.2f}\n"
            "Assinaturas: {sig}\n"
            "---".format(
                name=name,
                lo=lo_bpm,
                hi=hi_bpm,
                lp0=lp[0],
                lp1=lp[1],
                mp0=mp[0],
                mp1=mp[1],
                hp0=hp[0],
                hp1=hp[1],
                hr0=hr_lo,
                hr1=hr_hi,
                sig=m["signatures"],
            )
        )
    return "\n".join(lines)


PROMPT = """
Você é um especialista em música eletrônica. Classifique a faixa com base nas FEATURES abaixo.
As features vêm de três janelas (mid60, center30, last60) agregadas por mediana/média ponderada (mais peso no final).

REGRAS:
- Use APENAS um subgênero dentre CANDIDATES.
- Compare FEATURES com CANDIDATE_RULES (BPM, Bandas%, HP Ratio).
- Calcule internamente uma similaridade ponderada:
  - BPM (peso 0.45): maior se o BPM cair dentro da faixa do candidato (ou perto do centro).
  - Bandas% (peso 0.40): maior quanto mais low/mid/high_pct caírem nas faixas do candidato.
  - HP Ratio (peso 0.15): maior se dentro da faixa.
- Escolha o candidato de MAIOR similaridade.
- Só use 'Subgênero Não Identificado' se a similaridade final for muito baixa (ex.: < 0.40).

RESPOSTA: exatamente UMA linha, no formato:
Subgênero: <um dos CANDIDATES ou 'Subgênero Não Identificado'>

Observações internas (não exponha):
- Absorções: Bass House ← Electro House; Uplifting/Progressive Trance ← Vocal Trance; Psytrance ← Goa; Dubstep ← Riddim.
- Priorize o candidato com melhor aderência NUMÉRICA às faixas (BPM/bandas/hp_ratio).
"""


def call_gpt(features: Dict[str, float | int | None], candidates: List[str]) -> str:
    """Envia FEATURES + CANDIDATES + CANDIDATE_RULES ao LLM e retorna o texto da resposta."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    rules_text = format_rules_for_candidates(candidates)
    user_payload = {
        "FEATURES": features,
        "CANDIDATES": candidates,
        "CANDIDATE_RULES": rules_text,
    }
    messages = [
        {"role": "system", "content": PROMPT},
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

    return body["choices"][0]["message"]["content"].strip()


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="saasdj-backend (organizado v3)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v3"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Recebe um .mp3/.wav, extrai features e classifica.
    Retorno:
    {
      "bpm": <int|None>,
      "subgenero": <str>,
      "analise": "BPM=...; low%=...; mid%=...; high%=...; hp=...; onset=...; kick=...; cands=[...]; chosen=...; source=llm|fallback"
    }
    """
    try:
        # Validação simples
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 1) Carregar 3 janelas e extrair features agregadas
        windows = load_audio_windows(data)
        feats = extract_features_multi(windows)

        # 2) Selecionar candidatos por BPM
        bpm_val = feats.get("bpm")
        cands = candidates_by_bpm(bpm_val)

        # 3) Chamar GPT com candidatos e regras
        try:
            content = call_gpt(feats, cands)
        except Exception as e:
            # Falha na OpenAI — Fallback heurístico
            fb_sub = backend_fallback_best_candidate(feats, cands)
            bpm_int = int(round(bpm_val)) if bpm_val is not None else None

            decision_source = "fallback"
            tech_line = build_tech_line(
                feats=feats,
                cands=cands,
                chosen=fb_sub if fb_sub in SUBGENRES else "Subgênero Não Identificado",
                decision_source=decision_source,
            )

            return JSONResponse(
                status_code=502,
                content={
                    "bpm": bpm_int,
                    "subgenero": fb_sub if fb_sub in SUBGENRES else "Subgênero Não Identificado",
                    "analise": tech_line,
                    "error": str(e),
                },
            )

        # 4) Parse: "Subgênero: X"
        sub = "Subgênero Não Identificado"
        for line in content.splitlines():
            L = line.strip().lower()
            if L.startswith("subgênero:") or L.startswith("subgenero:") or L.startswith("subgénero:"):
                sub = line.split(":", 1)[1].strip()

        # 5) Sanitiza subgênero (apenas universo permitido)
        if sub != "Subgênero Não Identificado" and sub not in SOFT_RULES:
            sub = "Subgênero Não Identificado"

        # 6) Fallback heurístico se o LLM não identificar
        decision_source = "llm"
        if sub == "Subgênero Não Identificado":
            fb_sub = backend_fallback_best_candidate(feats, cands)
            if fb_sub != "Subgênero Não Identificado":
                sub = fb_sub
                decision_source = "fallback"

        # 7) BPM como INTEIRO na resposta (arredondado)
        bpm_out = int(round(bpm_val)) if bpm_val is not None else None

        # 8) SEMPRE incluir a linha técnica "analise"
        tech_line = build_tech_line(
            feats=feats,
            cands=cands,
            chosen=sub,
            decision_source=decision_source,
        )

        return {
            "bpm": bpm_out,
            "subgenero": sub,
            "analise": tech_line,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        # Erro inesperado: tenta enriquecer com o que for possível
        payload = {
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "error": f"processing failed: {e.__class__.__name__}: {e}",
        }
        try:
            bpm_val = locals().get("bpm_val", None)
            feats = locals().get("feats", None)
            cands = locals().get("cands", [])
            if feats:
                bpm_out = int(round(bpm_val)) if bpm_val is not None else None
                tech_line = build_tech_line(
                    feats=feats,
                    cands=cands if cands else list(SOFT_RULES.keys()),
                    chosen="Subgênero Não Identificado",
                    decision_source="fallback-error",
                )
                payload.update({"bpm": bpm_out, "analise": tech_line})
        except Exception:
            pass

        return JSONResponse(status_code=500, content=payload)
