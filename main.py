# -*- coding: utf-8 -*-
"""
saasdj-backend (organizado v4)

- FastAPI para classificar subgênero de música eletrônica a partir de .mp3/.wav
- Extrai features com librosa (BPM robusto com consenso de janelas, bandas espectrais,
  HP ratio, onset normalizado, kickband, duração)
- Usa regras numéricas + GPT para eleger o subgênero
- Resposta SEMPRE inclui: {"bpm": <int|None>, "subgenero": <str>, "analise": "<linha técnica>"}

Taxonomia atualizada:
- Melodic Techno -> Melodic House & Techno (guarda-chuva House/Techno melódico)
- Uplifting Trance removido; Progressive Trance -> Melodic & Progressive Trance
- Minimal Bass (Tech House) removido (absorvido por Tech House / Bass House)
- Future House removido (absorvido por Progressive EDM & Future House)
- Detroit/Acid/Industrial Techno unificados em Old School Techno (Detroit/Acid/Industrial)
- Adicionado Hard Dance/Groove
- Progressive EDM -> Progressive EDM & Future House

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
# Lista oficial de subgêneros (universo permitido) — ATUALIZADA
# =============================================================================

SUBGENRES: List[str] = [
    # House
    "Deep House",
    "Tech House",
    "Progressive House",
    "Bass House",
    "Funky / Soulful House",
    "Brazilian Bass",
    "Afro House",
    "Indie Dance",

    # Techno / crossover
    "Old School Techno (Detroit/Acid/Industrial)",
    "Peak Time Techno",
    "Hard Techno",
    "High-Tech Minimal",
    "Melodic House & Techno",

    # Trance
    "Melodic & Progressive Trance",
    "Psytrance",
    "Dark Psytrance",

    # EDM (Festival)
    "Big Room",
    "Progressive EDM & Future House",

    # Hard Dance
    "Hardstyle",
    "Rawstyle",
    "Gabber Hardcore",
    "UK/Happy Hardcore",
    "Jumpstyle",
    "Hard Dance/Groove",

    # Bass Music
    "Dubstep",
    "Drum & Bass",
    "Liquid DnB",
    "Neurofunk",
]


# =============================================================================
# SOFT RULES NUMÉRICAS (por subgênero) — ATUALIZADAS
# Bandas (proporções 0–1): low=20–250 Hz | mid=250–4000 Hz | high=4000–20000 Hz
# hp_ratio = H/P (harmônico ÷ percussivo); onset_strength = média normalizada do onset
# Observação: As regras servem como "faixas-alvo" flexíveis, não cortes duros.
# =============================================================================

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
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.12, 0.36)},
        "hp_ratio": (0.75, 1.25),
        "onset_strength": (0.40, 0.70),
        "signatures": "kick/bass secos e funcionais, grooves repetitivos, poucos leads",
    },
    "Progressive House": {
        # PH deve ser mais seletivo: tende a faixas longas e progressivas
        "bpm": (122, 128),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.40, 0.60), "high": (0.10, 0.28)},
        "hp_ratio": (1.10, 1.70),
        "onset_strength": (0.25, 0.55),
        "signatures": "builds longos, atmosfera melódica, progressão constante/emotiva (muitas vezes >5min)",
    },
    "Bass House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.45, 0.70), "mid": (0.20, 0.40), "high": (0.18, 0.35)},
        "hp_ratio": (0.80, 1.15),
        "onset_strength": (0.45, 0.75),
        "signatures": "basslines agressivas/‘talking’, queda forte no drop (absorve Electro House/minimal bass)",
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
    "Afro House": {
        "bpm": (118, 125),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.35, 0.55), "high": (0.12, 0.28)},
        "hp_ratio": (1.05, 2.20),
        "onset_strength": (0.35, 0.65),
        "signatures": "percussões afro, groove orgânico, vocais/texturas étnicas",
    },
    "Indie Dance": {
        # Foco: mid alto, high contido, kick discreto, onsets contidos, BPM mais baixo
        "bpm": (110, 123),
        "bands_pct": {"low": (0.18, 0.33), "mid": (0.48, 0.62), "high": (0.10, 0.32)},
        "hp_ratio": (1.30, 1.90),
        "onset_strength": (0.20, 0.55),
        "signatures": "vibe retrô/alternativa, synths vintage, menos ênfase em transientes/kick pesado",
    },

    # ---------------- TECHNO / CROSSOVER ----------------
    "Old School Techno (Detroit/Acid/Industrial)": {
        "bpm": (124, 138),
        "bands_pct": {"low": (0.28, 0.50), "mid": (0.28, 0.52), "high": (0.12, 0.32)},
        "hp_ratio": (0.85, 1.35),
        "onset_strength": (0.45, 0.75),
        "signatures": "estética clássica/analógica, 303/linhas repetitivas/industrial; menos ênfase épica",
    },
    "Peak Time Techno": {
        "bpm": (128, 136),
        "bands_pct": {"low": (0.32, 0.56), "mid": (0.24, 0.46), "high": (0.18, 0.35)},
        "hp_ratio": (0.85, 1.30),
        "onset_strength": (0.55, 0.85),
        "signatures": "4x4 direto para ápice; leads discretos; energia constante",
    },
    "Hard Techno": {
        "bpm": (135, 165),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.70, 1.30),
        "onset_strength": (0.65, 0.95),
        "signatures": "agressivo e percussivo; kicks duros; pouca melodia",
    },
    "High-Tech Minimal": {
        "bpm": (124, 130),
        "bands_pct": {"low": (0.30, 0.58), "mid": (0.22, 0.40), "high": (0.10, 0.25)},
        "hp_ratio": (0.85, 1.60),
        "onset_strength": (0.35, 0.65),
        "signatures": "minimalista, design sonoro detalhista; sub consistente, timbres enxutos",
    },
    "Melodic House & Techno": {
        "bpm": (120, 128),
        "bands_pct": {"low": (0.22, 0.42), "mid": (0.40, 0.62), "high": (0.15, 0.34)},
        "hp_ratio": (1.20, 2.00),
        "onset_strength": (0.30, 0.65),
        "signatures": "pads/leads melódicos e cinematográficos; crossover House/Techno (Artbat/Argy/Yubik/Monolink/WhoMadeWho)",
    },

    # ---------------- TRANCE ----------------
    "Melodic & Progressive Trance": {
        "bpm": (132, 140),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.42, 0.62), "high": (0.20, 0.36)},
        "hp_ratio": (1.20, 2.20),
        "onset_strength": (0.50, 0.90),
        "signatures": "trance melódico/atmosférico (inclui vocal/prog), builds expressivos",
    },
    "Psytrance": {
        "bpm": (138, 146),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.32, 0.52), "high": (0.15, 0.30)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.35, 0.65),
        "signatures": "rolling bass e FX psicodélicos; absorve Goa",
    },
    "Dark Psytrance": {
        "bpm": (145, 150),
        "bands_pct": {"low": (0.30, 0.50), "mid": (0.25, 0.45), "high": (0.20, 0.40)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.50, 0.80),
        "signatures": "mais escuro/denso; texturas e ruídos em destaque",
    },

    # ---------------- EDM (FESTIVAL) ----------------
    "Big Room": {
        "bpm": (126, 128),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.90, 1.40),
        "onset_strength": (0.60, 0.85),
        "signatures": "drops marcados; dinâmica alta; hi-hats/brilho dando impacto",
    },
    "Progressive EDM & Future House": {
        "bpm": (126, 130),
        "bands_pct": {"low": (0.25, 0.50), "mid": (0.40, 0.60), "high": (0.25, 0.45)},
        "hp_ratio": (1.40, 2.20),
        "onset_strength": (0.35, 0.65),
        "signatures": "melódico de festival ou future house polido/brilhante",
    },

    # ---------------- HARD DANCE ----------------
    "Hardstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.25, 0.45)},
        "hp_ratio": (0.80, 1.20),
        "onset_strength": (0.70, 0.98),
        "signatures": "kicks distorcidos/‘reverse bass’, intensidade alta",
    },
    "Rawstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.25, 0.48)},
        "hp_ratio": (0.75, 1.10),
        "onset_strength": (0.75, 0.99),
        "signatures": "mais agressivo/distorcido que Hardstyle; sound design extremo",
    },
    "Gabber Hardcore": {
        "bpm": (170, 190),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.20, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.70, 1.05),
        "onset_strength": (0.80, 0.99),
        "signatures": "muito agressivo, kicks ‘serra’, textura crua",
    },
    "UK/Happy Hardcore": {
        "bpm": (165, 180),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.35, 0.55), "high": (0.25, 0.45)},
        "hp_ratio": (1.00, 1.60),
        "onset_strength": (0.70, 0.95),
        "signatures": "rápido, eufórico/brilhante, melodias felizes",
    },
    "Jumpstyle": {
        "bpm": (140, 150),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.25, 0.45), "high": (0.15, 0.32)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.55, 0.80),
        "signatures": "padrões rítmicos ‘saltados’, batidas quadradas",
    },
    "Hard Dance/Groove": {
        "bpm": (135, 150),
        "bands_pct": {"low": (0.33, 0.58), "mid": (0.24, 0.44), "high": (0.18, 0.35)},
        "hp_ratio": (0.85, 1.30),
        "onset_strength": (0.60, 0.90),
        "signatures": "pegada hard porém dançante/pop-friendly (ex.: VTSS/Odymel), 4x4 forte, mas menos serrado que Raw/Hardstyle",
    },

    # ---------------- BASS MUSIC ----------------
    "Dubstep": {
        "bpm": (136, 146),
        "bands_pct": {"low": (0.45, 0.75), "mid": (0.20, 0.40), "high": (0.20, 0.40)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.60, 0.90),
        "signatures": "half-time ~140, wobble/sub pesado; absorve Riddim",
    },
    "Drum & Bass": {
        "bpm": (170, 180),
        "bands_pct": {"low": (0.30, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.75, 0.99),
        "signatures": "breakbeat rápido, muitos ataques, energia constante",
    },
    "Liquid DnB": {
        "bpm": (170, 176),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.38, 0.58), "high": (0.15, 0.30)},
        "hp_ratio": (1.05, 1.60),
        "onset_strength": (0.45, 0.75),
        "signatures": "suave/atmosférico, pads/vocais; break mais limpo",
    },
    "Neurofunk": {
        "bpm": (172, 178),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.65, 0.95),
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


def _measure_duration_sec(file_bytes: bytes) -> float | None:
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        return len(audio) / 1000.0
    except Exception:
        return None


def load_audio_windows(file_bytes: bytes, sr: int = 22050) -> Tuple[Dict[str, Tuple[np.ndarray, int]], float]:
    """
    Janela única 'mid90': 90s a partir de 60s (1:00 -> 2:30).
    Fallbacks:
      - se a faixa < 150s: usa 90s centrados (ou o máximo possível).
    Retorna (windows, duration_sec).
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_sec = len(audio) / 1000.0
    except Exception:
        audio = None
        duration_sec = None

    # alvo: offset=60.0, dur=90.0
    target_off, target_dur = 60.0, 90.0

    if duration_sec is None:
        # sem metadados de duração — tenta carregar a partir de 60s por 90s
        y, sr = _load_segment(file_bytes, target_off, target_dur, sr)
        return {"mid90": (y, sr)}, float(len(y) / sr)

    if duration_sec >= 150.0:
        off = target_off
        dur = min(target_dur, duration_sec - off)
    else:
        # se não tem 150s, pega 90s centrados, ou o que couber
        center = max(0.0, duration_sec / 2.0 - 45.0)
        off = max(0.0, min(center, max(0.0, duration_sec - 90.0)))
        dur = min(90.0, duration_sec - off)

    y, sr = _load_segment(file_bytes, off, dur, sr)
    return {"mid90": (y, sr)}, float(duration_sec)


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
    kick = float(yf[(xf >= 40) & (xf < 100)].sum())  # ajuda a diferenciar 4x4 de base forte

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


def _choose_bpm_consensus(window_bpms: List[float]) -> float | None:
    """
    Escolhe BPM por consenso das janelas.
    - Se 2/3 estão próximos (±2 BPM), escolher esse cluster.
    - Se há opção >150 e outra perto da metade (~x0.5) dentro de ±2 BPM, preferir a metade (anti-dobro).
    """
    if not window_bpms:
        return None
    arr = np.array([b for b in window_bpms if b is not None], dtype=float)
    if arr.size == 0:
        return None

    # Cluster aproximado por arredondamento
    rounded = np.round(arr).astype(int)
    # tentativa de cluster majoritário
    vals, counts = np.unique(rounded, return_counts=True)
    idx_max = np.argmax(counts)
    consensus = float(vals[idx_max])

    # anti-dobro: se consenso >150, verificar metade aproximada presente
    if consensus > 150:
        half = consensus / 2.0
        if np.any(np.abs(arr - half) <= 2.0):
            return float(np.median(arr[np.abs(arr - half) <= 2.0]))
    return consensus


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
        # leve prior anti >150 para estilos que não combinam com faixas/bandas
        if b > 150:
            s *= 0.98
        if s > best_sum:
            best_sum = s
            best_bpm = b
    return best_bpm


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float | int | None]:
    """Features de UMA janela."""
    # BPM robusto (candidatos)
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


def extract_features_multi(windows: Dict[str, Tuple[np.ndarray, int]], duration_sec: float) -> Dict[str, float | int | None]:
    """
    Extrai features apenas da janela 'mid90'.
    Inclui 'duration_sec' nas features para regras baseadas em duração.
    """
    y, sr = windows["mid90"]
    f = extract_features(y, sr)
    f["duration_sec"] = float(duration_sec)
    return f


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


def _indie_dance_bonus(lp, mp, hpv, hpr, onset, bpm, kick, duration_sec) -> float:
    """
    Reforça Indie Dance quando assinatura bate nos critérios-chave:
    - BPM <= 123 (preferencialmente <=122)
    - mid alto, high contido, low moderado
    - kick discreto, onsets contidos, hp_ratio elevado
    """
    score = 1.0
    if bpm is None:
        return score
    if bpm <= 123 and mp is not None and hpv is not None and lp is not None:
        ok_mid = mp >= 0.48
        ok_high = hpv <= 0.32
        ok_low = lp <= 0.33
        ok_onset = (onset is not None and onset <= 0.55)
        ok_hp = (hpr is not None and hpr >= 1.30)
        # kick discreto relativo ao low
        ok_kick = False
        if kick is not None:
            # aproximar uma razão "kick vs low": se low > 0, testa; senão, ignora
            ok_kick = (lp > 0.0 and kick > 0.0 and (kick / (1e-6 + kick + 0.5 * (1e-6 + 1.0))) < 0.50) or (kick < 1e7)
        if ok_mid and ok_high and ok_low and ok_onset and ok_hp:
            score *= 1.05
            if bpm <= 122:
                score *= 1.05
            if ok_kick:
                score *= 1.03
            # se duração for longa E progressiva poderia puxar para PH; aqui não bonificamos mais
    return score


def backend_fallback_best_candidate(features: Dict[str, float | int | None], candidates: List[str]) -> str:
    """Se o LLM falhar, escolhe o melhor candidato (sem confidence). Aplica viés por duração p/ Progressive House."""
    bpm = features.get("bpm")
    lp = features.get("low_pct", 0.0) or 0.0
    mp = features.get("mid_pct", 0.0) or 0.0
    hpv = features.get("high_pct", 0.0) or 0.0
    hpr = features.get("hp_ratio", 0.0) or 0.0
    kick = features.get("kick_40_100", 0.0) or 0.0
    dur = float(features.get("duration_sec", 0.0) or 0.0)

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
        s_bpm  = _score_in_range(bpm, rule["bpm"])
        s_low  = _score_in_range(lp,  rule["bands_pct"]["low"])
        s_mid  = _score_in_range(mp,  rule["bands_pct"]["mid"])
        s_high = _score_in_range(hpv, rule["bands_pct"]["high"])
        s_hp   = _score_in_range(hpr, rule["hp_ratio"])

        bands_avg = (s_low + s_mid + s_high) / 3.0
        score = 0.45 * s_bpm + 0.40 * bands_avg + 0.15 * s_hp

        # bônus suave para pista 4x4 com kick forte
        if name in ("Tech House", "Peak Time Techno", "Hard Techno", "High-Tech Minimal") and kick > 0:
            score *= 1.03

        # viés forte por duração para Progressive House
        if name == "Progressive House":
            if dur >= 300.0:   # 5 minutos
                score *= 1.20   # bônus forte
            else:
                score *= 0.80   # penalidade forte

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
As features vêm de uma janela única de 90s entre 60s e 150s (1:00 → 2:30), pensada para capturar o trecho mais representativo.

REGRAS:
- Use APENAS um subgênero dentre CANDIDATES.
- Compare FEATURES com CANDIDATE_RULES (BPM, Bandas%, HP Ratio).
- Calcule internamente uma similaridade ponderada:
  - BPM (peso 0.45)
  - Bandas% (peso 0.40)
  - HP Ratio (peso 0.15)
- Duração como heurística adicional:
  - Se DURATION_SEC ≥ 300s (≥ 5 min), aplique forte viés pró "Progressive House" quando BPM e Bandas% forem compatíveis.
  - Se DURATION_SEC < 300s, penalize "Progressive House".
- Escolha o candidato de MAIOR similaridade.
- Só use 'Subgênero Não Identificado' se a similaridade final for muito baixa (ex.: < 0.40).

RESPOSTA: exatamente UMA linha, no formato:
Subgênero: <um dos CANDIDATES ou 'Subgênero Não Identificado'>

Observações internas (não exponha):
- Absorções (se aplicável): Bass House ← Electro House; Psytrance ← Goa; Dubstep ← Riddim.
- Priorize a aderência NUMÉRICA às faixas (BPM/bandas/hp_ratio) + heurística de duração descrita acima.
"""



def call_gpt(
    features_agg: Dict[str, float | int | None],
    features_windows: Dict[str, Dict[str, float | int | None]],
    candidates: List[str],
    duration_sec: float | None,
) -> str:
    """Envia FEATURES (agregadas e por janela) + duração + CANDIDATES + CANDIDATE_RULES ao LLM e retorna o texto da resposta."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    rules_text = format_rules_for_candidates(candidates)
    user_payload = {
        "FEATURES_AGG": features_agg,
        "FEATURES_WINDOWS": features_windows,  # dá contexto ao LLM sobre trechos (melódico vs pista)
        "DURATION_SEC": duration_sec,
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

app = FastAPI(title="saasdj-backend (organizado v4)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v4"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Recebe um .mp3/.wav, extrai features e classifica.
    Retorno:
    {
      "bpm": <int|None>,
      "subgenero": <str>,
      "analise": "BPM=...; low%=...; mid%=...; high%=...; hp=...; onset=...; kick=...; dur=...s; cands=[...]; chosen=...; source=llm|fallback"
    }
    """
    try:
        # Validação simples
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 0) Duração total
        duration_sec = _measure_duration_sec(data)

        # 1) Carregar janela única (mid90) e extrair features
        windows, duration_sec = load_audio_windows(data)
        feats = extract_features_multi(windows, duration_sec)

        # 1.1) Reforço anti-dobro (consenso já considerado; aqui uma última checagem)
        bpm_val = feats_agg.get("bpm")
        if bpm_val is not None and bpm_val > 150:
            half = bpm_val / 2.0
            # heurística: se half cai bem nas faixas de House/Techno/Trance e bandas não parecem Hard,
            # usar half como BPM final
            friendly = ("Tech House", "Progressive House", "Melodic House & Techno", "Peak Time Techno", "Afro House", "Indie Dance",
                        "Old School Techno (Detroit/Acid/Industrial)", "Big Room", "Progressive EDM & Future House")
            ok_half = False
            for name in friendly:
                lo, hi = SOFT_RULES[name]["bpm"]
                if (half >= lo - 2) and (half <= hi + 2):
                    ok_half = True
                    break
            # se high% e onset não são de Hard Dance, preferir half
            if ok_half:
                if (feats_agg.get("high_pct", 0) <= 0.35) and (feats_agg.get("onset_strength", 0) <= 0.85):
                    feats_agg["bpm"] = round(half, 3)

        # 2) Selecionar candidatos por BPM
        bpm_val = feats_agg.get("bpm")
        cands = candidates_by_bpm(bpm_val)

        # 3) Chamar GPT com candidatos e regras
        try:
            content = call_gpt(feats_agg, feats_windows, cands, duration_sec)
        except Exception as e:
            # Falha na OpenAI — Fallback heurístico
            fb_sub = backend_fallback_best_candidate(feats_agg, cands, duration_sec)
            bpm_int = int(round(bpm_val)) if bpm_val is not None else None

            decision_source = "fallback"
            tech_line = build_tech_line(
                feats=feats_agg,
                cands=cands,
                chosen=fb_sub if fb_sub in SUBGENRES else "Subgênero Não Identificado",
                decision_source=decision_source,
                duration_sec=duration_sec,
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
            fb_sub = backend_fallback_best_candidate(feats_agg, cands, duration_sec)
            if fb_sub != "Subgênero Não Identificado":
                sub = fb_sub
                decision_source = "fallback"

        # 7) BPM como INTEIRO na resposta (arredondado)
        bpm_out = int(round(bpm_val)) if bpm_val is not None else None

        # 8) SEMPRE incluir a linha técnica "analise"
        tech_line = build_tech_line(
            feats=feats_agg,
            cands=cands,
            chosen=sub,
            decision_source=decision_source,
            duration_sec=duration_sec,
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
            duration_sec = locals().get("duration_sec", None)
            feats = locals().get("feats_agg", None)
            cands = locals().get("cands", [])
            if feats:
                bpm_val = feats.get("bpm")
                bpm_out = int(round(bpm_val)) if bpm_val is not None else None
                tech_line = build_tech_line(
                    feats=feats,
                    cands=cands if cands else list(SOFT_RULES.keys()),
                    chosen="Subgênero Não Identificado",
                    decision_source="fallback-error",
                    duration_sec=duration_sec,
                )
                payload.update({"bpm": bpm_out, "analise": tech_line})
        except Exception:
            pass

        return JSONResponse(status_code=500, content=payload)
