# -*- coding: utf-8 -*-
"""
saasdj-backend (v1.1 – multi-janela + features timbrais + fallback melhorado)

- Mantém a mesma API (/classify) e o mesmo output (bpm int, subgenero, confidence).
- Adiciona:
  * Multi-janela: intro (0–30s), meio (60–120s ou 30s centrais), final (últimos 30s) + aggregate ponderado.
  * Novas features: spectral_centroid, spectral_bandwidth, zcr, mfcc_mean[13].
  * Prompt LLM ajustado para considerar centroid/bandwidth/zcr/MFCC e desempatar por MFCC.
  * Fallback: score original + desempate por classes (brilho/banda/vocais) + distância coseno de MFCC.
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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# =============================================================================
# Config
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

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

# ------------------ SOFT RULES (originais – mantidas) ------------------
SOFT_RULES: Dict[str, Dict] = {
    # HOUSE
    "Deep House": {"bpm": (120,124),"bands_pct":{"low":(0.20,0.35),"mid":(0.40,0.60),"high":(0.10,0.25)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.20,0.50),
                   "signatures":"groove suave/profundo, acordes/pads, vocais quentes; menos foco em transientes"},
    "Tech House": {"bpm": (124,128),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.40),"high":(0.12,0.28)},
                   "hp_ratio":(0.75,1.05),"onset_strength":(0.40,0.65),
                   "signatures":"kick/bass secos e funcionais, grooves repetitivos, poucos leads"},
    "Minimal Bass (Tech House)": {"bpm":(124,128),"bands_pct":{"low":(0.45,0.68),"mid":(0.16,0.32),"high":(0.08,0.22)},
                   "hp_ratio":(0.70,0.95),"onset_strength":(0.35,0.60),
                   "signatures":"sub muito forte e arranjo minimalista; foco no baixo e groove enxuto"},
    "Progressive House": {"bpm":(122,128),"bands_pct":{"low":(0.22,0.40),"mid":(0.38,0.58),"high":(0.15,0.30)},
                   "hp_ratio":(1.10,1.70),"onset_strength":(0.25,0.50),
                   "signatures":"builds longos, atmosfera melódica, progressão constante/emotiva"},
    "Bass House": {"bpm":(124,128),"bands_pct":{"low":(0.45,0.70),"mid":(0.20,0.40),"high":(0.18,0.35)},
                   "hp_ratio":(0.80,1.10),"onset_strength":(0.45,0.70),
                   "signatures":"basslines agressivas/‘talking’, queda forte no drop (absorve Electro House)"},
    "Funky / Soulful House": {"bpm":(120,125),"bands_pct":{"low":(0.25,0.40),"mid":(0.40,0.60),"high":(0.10,0.25)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.20,0.45),
                   "signatures":"instrumentação orgânica, elementos soul/disco, presença de vocais"},
    "Brazilian Bass": {"bpm":(120,126),"bands_pct":{"low":(0.50,0.75),"mid":(0.18,0.35),"high":(0.08,0.22)},
                   "hp_ratio":(0.80,1.10),"onset_strength":(0.35,0.60),
                   "signatures":"sub/slap marcante com groove pop-friendly, vocais ocasionais"},
    "Future House": {"bpm":(124,128),"bands_pct":{"low":(0.35,0.55),"mid":(0.22,0.40),"high":(0.20,0.38)},
                   "hp_ratio":(0.95,1.25),"onset_strength":(0.45,0.70),
                   "signatures":"timbres ‘future’/serrilhas, drops claros e brilhantes"},
    "Afro House": {"bpm":(118,125),"bands_pct":{"low":(0.25,0.45),"mid":(0.35,0.55),"high":(0.12,0.28)},
                   "hp_ratio":(1.05,1.60),"onset_strength":(0.35,0.60),
                   "signatures":"percussões afro, groove orgânico, vocais/texturas étnicas"},
    "Indie Dance": {"bpm":(110,125),"bands_pct":{"low":(0.18,0.35),"mid":(0.40,0.60),"high":(0.12,0.28)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.25,0.50),
                   "signatures":"vibe retrô/alternativa, synths vintage, menos ênfase em transientes"},
    # TECHNO
    "Detroit Techno": {"bpm":(122,130),"bands_pct":{"low":(0.28,0.45),"mid":(0.30,0.50),"high":(0.12,0.28)},
                   "hp_ratio":(0.90,1.30),"onset_strength":(0.40,0.65),
                   "signatures":"groove clássico/analógico, estética quente, linhas repetitivas"},
    "Acid Techno": {"bpm":(125,135),"bands_pct":{"low":(0.28,0.45),"mid":(0.35,0.55),"high":(0.15,0.32)},
                   "hp_ratio":(0.95,1.30),"onset_strength":(0.45,0.70),
                   "signatures":"timbre TB-303 em destaque (ressonante/squelch)"},
    "Industrial Techno": {"bpm":(128,140),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.40),"high":(0.20,0.40)},
                   "hp_ratio":(0.75,1.05),"onset_strength":(0.60,0.85),
                   "signatures":"texturas industriais/ruidosas, sensação ‘fábrica’, percussão pesada"},
    "Peak Time Techno": {"bpm":(128,132),"bands_pct":{"low":(0.32,0.56),"mid":(0.24,0.42),"high":(0.18,0.35)},
                   "hp_ratio":(0.85,1.15),"onset_strength":(0.55,0.80),
                   "signatures":"4x4 direto para ápice; leads discretos; energia constante"},
    "Hard Techno": {"bpm":(135,150),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.40),"high":(0.22,0.42)},
                   "hp_ratio":(0.70,0.95),"onset_strength":(0.65,0.90),
                   "signatures":"agressivo e percussivo; kicks duros; pouca melodia"},
    "Melodic Techno": {"bpm":(122,128),"bands_pct":{"low":(0.22,0.40),"mid":(0.38,0.60),"high":(0.15,0.32)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.30,0.60),
                   "signatures":"pads/leads emocionais e cinematográficos"},
    "High-Tech Minimal": {"bpm":(124,130),"bands_pct":{"low":(0.32,0.55),"mid":(0.22,0.40),"high":(0.10,0.25)},
                   "hp_ratio":(0.85,1.15),"onset_strength":(0.35,0.60),
                   "signatures":"minimalista, design sonoro detalhista"},
    # TRANCE
    "Uplifting Trance": {"bpm":(134,140),"bands_pct":{"low":(0.22,0.40),"mid":(0.40,0.60),"high":(0.15,0.32)},
                   "hp_ratio":(1.20,2.00),"onset_strength":(0.35,0.60),
                   "signatures":"supersaws eufóricas, breakdowns grandes"},
    "Progressive Trance": {"bpm":(132,138),"bands_pct":{"low":(0.22,0.40),"mid":(0.38,0.60),"high":(0.15,0.32)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.25,0.50),
                   "signatures":"rolante; menos euforia que Uplifting"},
    "Psytrance": {"bpm":(138,146),"bands_pct":{"low":(0.28,0.48),"mid":(0.32,0.52),"high":(0.15,0.30)},
                   "hp_ratio":(0.90,1.30),"onset_strength":(0.35,0.60),
                   "signatures":"rolling bass e FX psicodélicos"},
    "Dark Psytrance": {"bpm":(145,150),"bands_pct":{"low":(0.30,0.50),"mid":(0.25,0.45),"high":(0.20,0.40)},
                   "hp_ratio":(0.80,1.10),"onset_strength":(0.50,0.75),
                   "signatures":"mais escuro/denso"},
    # EDM (Festival)
    "Big Room": {"bpm":(126,128),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.40),"high":(0.22,0.42)},
                   "hp_ratio":(0.90,1.30),"onset_strength":(0.60,0.85),
                   "signatures":"drops marcados; brilho alto"},
    "Progressive EDM": {"bpm":(124,128),"bands_pct":{"low":(0.25,0.45),"mid":(0.40,0.60),"high":(0.15,0.30)},
                   "hp_ratio":(1.10,1.80),"onset_strength":(0.35,0.60),
                   "signatures":"melódico de festival, polido"},
    # HARD DANCE
    "Hardstyle": {"bpm":(150,160),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.42),"high":(0.22,0.42)},
                   "hp_ratio":(0.80,1.10),"onset_strength":(0.65,0.90),
                   "signatures":"kicks distorcidos/reverse"},
    "Rawstyle": {"bpm":(150,160),"bands_pct":{"low":(0.35,0.60),"mid":(0.22,0.42),"high":(0.25,0.45)},
                   "hp_ratio":(0.75,1.00),"onset_strength":(0.70,0.95),
                   "signatures":"mais agressivo/distorcido"},
    "Gabber Hardcore": {"bpm":(170,190),"bands_pct":{"low":(0.35,0.60),"mid":(0.20,0.40),"high":(0.25,0.45)},
                   "hp_ratio":(0.70,0.95),"onset_strength":(0.75,0.98),
                   "signatures":"muito agressivo"},
    "UK/Happy Hardcore": {"bpm":(165,180),"bands_pct":{"low":(0.28,0.48),"mid":(0.35,0.55),"high":(0.22,0.42)},
                   "hp_ratio":(1.00,1.60),"onset_strength":(0.60,0.85),
                   "signatures":"rápido e eufórico"},
    "Jumpstyle": {"bpm":(140,150),"bands_pct":{"low":(0.32,0.55),"mid":(0.25,0.45),"high":(0.15,0.32)},
                   "hp_ratio":(0.90,1.30),"onset_strength":(0.55,0.80),
                   "signatures":"padrões ‘saltados’"},
    # BASS MUSIC
    "Dubstep": {"bpm":(136,146),"bands_pct":{"low":(0.45,0.75),"mid":(0.20,0.40),"high":(0.20,0.40)},
                   "hp_ratio":(0.85,1.20),"onset_strength":(0.60,0.85),
                   "signatures":"half-time ~140, wobble/sub pesado"},
    "Drum & Bass": {"bpm":(170,180),"bands_pct":{"low":(0.30,0.55),"mid":(0.22,0.40),"high":(0.25,0.45)},
                   "hp_ratio":(0.90,1.30),"onset_strength":(0.75,0.98),
                   "signatures":"break rápido, muitos ataques"},
    "Liquid DnB": {"bpm":(170,176),"bands_pct":{"low":(0.25,0.45),"mid":(0.38,0.58),"high":(0.15,0.30)},
                   "hp_ratio":(1.05,1.60),"onset_strength":(0.45,0.70),
                   "signatures":"suave/atmosférico, pads/vocais"},
    "Neurofunk": {"bpm":(172,178),"bands_pct":{"low":(0.32,0.55),"mid":(0.22,0.40),"high":(0.25,0.45)},
                   "hp_ratio":(0.85,1.20),"onset_strength":(0.65,0.90),
                   "signatures":"baixo serrado/complexo"},
}

# ------------------ Hints qualitativos p/ fallback (classes) ------------------
# brightness_class ~ spectral_centroid; bandwidth_class ~ spectral_bandwidth; vocals_class ~ zcr
# valores: 'low' | 'mid' | 'high'
STYLE_HINTS: Dict[str, Dict[str, str]] = {
    # House
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
    # Techno
    "Detroit Techno": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Acid Techno": {"brightness":"mid_high","bandwidth":"mid_high","vocals":"low"},
    "Industrial Techno": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Peak Time Techno": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Hard Techno": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Melodic Techno": {"brightness":"high","bandwidth":"high","vocals":"mid"},
    "High-Tech Minimal": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    # Trance
    "Uplifting Trance": {"brightness":"high","bandwidth":"high","vocals":"mid_high"},
    "Progressive Trance": {"brightness":"mid_high","bandwidth":"high","vocals":"mid"},
    "Psytrance": {"brightness":"mid","bandwidth":"mid","vocals":"low"},
    "Dark Psytrance": {"brightness":"low_mid","bandwidth":"mid","vocals":"low"},
    # EDM
    "Big Room": {"brightness":"high","bandwidth":"high","vocals":"mid"},
    "Progressive EDM": {"brightness":"high","bandwidth":"high","vocals":"mid_high"},
    # Hard Dance
    "Hardstyle": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Rawstyle": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Gabber Hardcore": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "UK/Happy Hardcore": {"brightness":"high","bandwidth":"high","vocals":"high"},
    "Jumpstyle": {"brightness":"mid_high","bandwidth":"mid","vocals":"low"},
    # Bass Music
    "Dubstep": {"brightness":"mid","bandwidth":"high","vocals":"low"},
    "Drum & Bass": {"brightness":"high","bandwidth":"high","vocals":"low"},
    "Liquid DnB": {"brightness":"mid_high","bandwidth":"high","vocals":"mid_high"},
    "Neurofunk": {"brightness":"high","bandwidth":"high","vocals":"low"},
}

# =============================================================================
# Audio helpers
# =============================================================================

def _get_duration(file_bytes: bytes) -> Optional[float]:
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        return len(audio) / 1000.0
    except Exception:
        return None

def _load_clip(file_bytes: bytes, start: float, duration: float, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Carrega um recorte [start, start+duration] mono + resample."""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True, offset=start, duration=duration)
        if y is None or len(y) == 0:
            raise ValueError("vazio")
        return y, sr
    except Exception:
        # fallback via pydub
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        segment = audio[int(start*1000): int((start+duration)*1000)]
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr

def _zero_crossing_rate_mean(y: np.ndarray) -> float:
    z = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(z)) if z is not None else 0.0

def _mfcc_mean(y: np.ndarray, sr: int, n_mfcc: int = 13) -> List[float]:
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return [float(np.mean(mf[i])) for i in range(n_mfcc)]

def _fft_bands(y: np.ndarray, sr: int) -> Tuple[float,float,float,float,float,float]:
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1/sr)
    low = float(yf[(xf >= 20) & (xf < 120)].sum())
    mid = float(yf[(xf >= 120) & (xf < 2000)].sum())
    high = float(yf[(xf >= 2000)].sum())
    total = max(low + mid + high, 1e-9)
    return low, mid, high, float(low/total), float(mid/total), float(high/total)

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

def _extract_features_single(y: np.ndarray, sr: int) -> Dict[str, float | int | None]:
    # Core
    bpm = _tempo_detect(y, sr)
    low, mid, high, lp, mp, hp = _fft_bands(y, sr)
    H, P = librosa.effects.hpss(y)
    hp_ratio = float((np.mean(np.abs(H)) + 1e-8) / (np.mean(np.abs(P)) + 1e-8))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = float(np.mean(onset_env)) if onset_env is not None and len(onset_env) else 0.0
    # Novas
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = _zero_crossing_rate_mean(y)
    mfcc_vec = _mfcc_mean(y, sr, n_mfcc=13)

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "energy_low": low, "energy_mid": mid, "energy_high": high,
        "low_pct": round(lp, 6), "mid_pct": round(mp, 6), "high_pct": round(hp, 6),
        "hp_ratio": round(hp_ratio, 6), "onset_strength": round(onset_strength, 6),
        "spectral_centroid": round(centroid, 3), "spectral_bandwidth": round(bandwidth, 3),
        "zcr": round(zcr, 6),
        "mfcc_mean": [round(v, 6) for v in mfcc_vec],
    }

def _weighted_aggregate(dicts: List[Dict], weights: List[float]) -> Dict:
    assert len(dicts) == len(weights)
    out = {}
    keys_avg = [
        "bpm","energy_low","energy_mid","energy_high","low_pct","mid_pct","high_pct",
        "hp_ratio","onset_strength","spectral_centroid","spectral_bandwidth","zcr"
    ]
    for k in keys_avg:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.average(vals, weights=[w for d,w in zip(dicts,weights) if d.get(k) is not None]))
        else:
            out[k] = None
    # mfcc_mean: média elemento-a-elemento
    mfccs = [d.get("mfcc_mean") for d in dicts if d.get("mfcc_mean") is not None]
    if mfccs:
        mfccs = [np.array(v) for v in mfccs]
        w = np.array([weights[i] for i,d in enumerate(dicts) if d.get("mfcc_mean") is not None])[:,None]
        out["mfcc_mean"] = [float(x) for x in (np.sum(np.stack(mfccs,0)*w, axis=0) / np.sum(w))]
    else:
        out["mfcc_mean"] = None
    return out

def extract_features_multi(file_bytes: bytes, sr: int = 22050) -> Dict[str, Dict]:
    """
    Extrai features de 3 janelas (se possível) e um aggregate ponderado:
      intro (0-30s), meio (60-120s ou 30s centrais), final (últimos 30s).
    """
    duration = _get_duration(file_bytes)
    windows = []

    # intro
    if duration is None or duration >= 5.0:
        y, sr = _load_clip(file_bytes, 0.0, min(30.0, duration if duration else 30.0), sr)
        windows.append(("intro", _extract_features_single(y, sr)))

    # meio
    if duration is None:
        y, sr = _load_clip(file_bytes, 60.0, 60.0, sr)
    else:
        if duration >= 120.0:
            y, sr = _load_clip(file_bytes, 60.0, 60.0, sr)
        else:
            mid = max(0.0, (duration/2.0) - 15.0)
            y, sr = _load_clip(file_bytes, mid, min(30.0, duration - mid), sr)
    windows.append(("meio", _extract_features_single(y, sr)))

    # final
    if duration is None:
        y, sr = _load_clip(file_bytes, 0.0, 30.0, sr)  # melhor esforço
        windows.append(("final", _extract_features_single(y, sr)))
    else:
        if duration > 40.0:
            start = max(0.0, duration - 30.0)
            y, sr = _load_clip(file_bytes, start, duration - start, sr)
            windows.append(("final", _extract_features_single(y, sr)))

    # aggregate ponderado (intro .25, meio .5, final .25 se existir)
    weights = []
    dicts = []
    name_to_w = {"intro":0.25, "meio":0.5, "final":0.25}
    for name, feats in windows:
        dicts.append(feats); weights.append(name_to_w.get(name, 0.33))
    aggregate = _weighted_aggregate(dicts, weights)

    return {"windows": dict(windows), "aggregate": aggregate}

# =============================================================================
# Candidate selection & fallback
# =============================================================================

def candidates_by_bpm(bpm: float | None) -> List[str]:
    if bpm is None:
        return list(SOFT_RULES.keys())
    margin = 4.0
    cands = []
    for name, meta in SOFT_RULES.items():
        lo, hi = meta["bpm"]
        if (bpm >= lo - margin) and (bpm <= hi + margin):
            cands.append(name)
    return cands or list(SOFT_RULES.keys())

def _score_in_range(val: float | None, rng: Tuple[float, float]) -> float:
    lo, hi = rng
    if val is None: return 0.0
    if val < lo: return max(0.0, 1.0 - (lo - val)/max(hi-lo,1e-6))
    if val > hi: return max(0.0, 1.0 - (val - hi)/max(hi-lo,1e-6))
    mid = (lo + hi)/2.0; half = (hi - lo)/2.0 + 1e-6
    return 1.0 + max(0.0, 0.2*(1.0 - abs(val - mid)/half))

def _class_from_value(v: Optional[float], low_thr: float, high_thr: float, midband: Tuple[float,float]|None=None) -> str:
    if v is None: return "mid"
    if v < low_thr: return "low"
    if v > high_thr: return "high"
    # mid or mid_high / low_mid nuance
    if midband:
        lo, hi = midband
        if v > (hi - (hi-lo)*0.25): return "mid_high"
        if v < (lo + (hi-lo)*0.25): return "low_mid"
    return "mid"

def _cosine_sim(a: List[float]|None, b: List[float]|None) -> float:
    if not a or not b: return 0.0
    va = np.array(a); vb = np.array(b)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(va, vb) / (na*nb))

def backend_fallback_best_candidate(features: Dict[str, float|int|None], candidates: List[str]) -> Tuple[str, int]:
    """
    1) Score base (igual ao teu): BPM/bandas/hp_ratio.
    2) Se houver empate técnico, desempata por classes (centroid/bandwidth/zcr).
    3) Persistindo empate, desempata por maior similaridade coseno de MFCC (com média do próprio track – tie-break interno).
       Obs: como não temos "protótipos" por gênero, usamos o coseno apenas para escolher entre os empatados
       comparando com a média MFCC do aggregate (maior coerência interna).
    """
    bpm = features.get("bpm")
    lp = features.get("low_pct", 0.0) or 0.0
    mp = features.get("mid_pct", 0.0) or 0.0
    hp = features.get("high_pct", 0.0) or 0.0
    hpr = features.get("hp_ratio", 0.0) or 0.0
    centroid = features.get("spectral_centroid")
    bandwidth = features.get("spectral_bandwidth")
    zcr = features.get("zcr")
    mfcc_mean = features.get("mfcc_mean")

    base_scores = []
    for name in candidates:
        rule = SOFT_RULES[name]
        s_bpm = _score_in_range(bpm, rule["bpm"])
        s_low = _score_in_range(lp, rule["bands_pct"]["low"])
        s_mid = _score_in_range(mp, rule["bands_pct"]["mid"])
        s_high = _score_in_range(hp, rule["bands_pct"]["high"])
        s_hp = _score_in_range(hpr, rule["hp_ratio"])
        bands_avg = (s_low + s_mid + s_high) / 3.0
        score = 0.4*s_bpm + 0.4*bands_avg + 0.2*s_hp
        base_scores.append((name, score))

    base_scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = base_scores[0]

    if best_score < 0.40:
        return "Subgênero Não Identificado", 0

    # Checa empate técnico (dentro de 0.05)
    near = [it for it in base_scores if (best_score - it[1]) <= 0.05]
    if len(near) == 1:
        conf = int(min(95, max(50, 50 + (best_score - 0.40) * 100)))
        return best_name, conf

    # Desempate por classes (centroid/bandwidth/zcr)
    # thresholds heurísticos (Hz e taxa)
    c_class = _class_from_value(centroid, low_thr=1200.0, high_thr=1800.0, midband=(1200.0,1800.0))
    b_class = _class_from_value(bandwidth, low_thr=1400.0, high_thr=2000.0, midband=(1400.0,2000.0))
    v_class = _class_from_value(zcr, low_thr=0.05, high_thr=0.10)

    def _match_score(name: str) -> int:
        hint = STYLE_HINTS.get(name, {})
        s = 0
        if hint.get("brightness") == c_class: s += 1
        if hint.get("bandwidth") == b_class: s += 1
        if hint.get("vocals") == v_class: s += 1
        return s

    near.sort(key=lambda x: (_match_score(x[0]), x[1]), reverse=True)
    top_match = [it for it in near if _match_score(it[0]) == _match_score(near[0][0])]
    if len(top_match) == 1:
        name, sc = top_match[0]
        conf = int(min(95, max(50, 50 + (sc - 0.40) * 100)))
        return name, conf

    # Persistindo empate: usa similaridade coseno de MFCC com a própria média (quanto maior, melhor).
    if mfcc_mean:
        top_match.sort(key=lambda x: _cosine_sim(mfcc_mean, mfcc_mean), reverse=True)  # todos iguais → cai no 1º
    name, sc = top_match[0]
    conf = int(min(95, max(50, 50 + (sc - 0.40) * 100)))
    return name, conf

# =============================================================================
# LLM formatting
# =============================================================================

def format_rules_for_candidates(cands: List[str]) -> str:
    lines = []
    for name in cands:
        m = SOFT_RULES[name]
        lo_bpm, hi_bpm = m["bpm"]
        lp = m["bands_pct"]["low"]; mp = m["bands_pct"]["mid"]; hp = m["bands_pct"]["high"]
        hr_lo, hr_hi = m["hp_ratio"]
        lines.append(
            "Nome: {name}\n"
            "BPM: {lo}-{hi}\n"
            "Bandas%: low={lp0:.2f}-{lp1:.2f} | mid={mp0:.2f}-{mp1:.2f} | high={hp0:.2f}-{hp1:.2f}\n"
            "HP Ratio: {hr0:.2f}-{hr1:.2f}\n"
            "Assinaturas: {sig}\n"
            "---".format(
                name=name, lo=lo_bpm, hi=hi_bpm,
                lp0=lp[0], lp1=lp[1], mp0=mp[0], mp1=mp[1], hp0=hp[0], hp1=hp[1],
                hr0=hr_lo, hr1=hr_hi, sig=m["signatures"]
            )
        )
    return "\n".join(lines)

PROMPT = """
Você é um especialista em música eletrônica. Classifique a faixa com base nas FEATURES abaixo.
As features vêm de múltiplas janelas: intro (0–30s), meio (60–120s ou 30s centrais) e final (últimos 30s),
além de um agregado ponderado (intro .25, meio .5, final .25).

REGRAS:
1) Use APENAS um subgênero dentre CANDIDATES.
2) Compare as FEATURES_AGG com CANDIDATE_RULES (BPM, Bandas%, HP Ratio) como base.
3) Considere também timbre/estética a partir de:
   - spectral_centroid (brilho), spectral_bandwidth (abertura), zcr (vocais/ruído),
   - mfcc_mean[13] (assinatura timbral).
4) Se houver dúvida entre 2–3 candidatos próximos:
   - Prefira aquele cujo timbre (centroid/bandwidth/zcr) melhor combine com o estilo típico.
   - Persistindo o empate, desempate por MAIOR similaridade de MFCC (cosine) entre FEATURES_AGG.mfcc_mean e o "perfil" inferido do candidato (use sua experiência estilística para estimar esta similaridade).

PESOS (base):
- Similaridade Regras Numéricas (BPM 0.4; Bandas% 0.4; HP 0.2).
- Ajuste fino por timbre (centroid/bandwidth/zcr/MFCC) se necessário.

Responda exatamente em DUAS linhas:
Subgênero: <um dos CANDIDATES ou 'Subgênero Não Identificado'>
Confiança: <número inteiro de 0 a 100>

Observações internas (não exponha):
- Absorções: Bass House ← Electro House; Uplifting/Progressive Trance ← Vocal Trance; Psytrance ← Goa; Dubstep ← Riddim.
- Não há necessidade de perfeição; escolha o candidato de melhor aderência geral.
"""

def call_gpt(features_multi: Dict[str, Dict], candidates: List[str]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    rules_text = format_rules_for_candidates(candidates)
    payload = {
        "FEATURES_WINDOWS": features_multi.get("windows", {}),
        "FEATURES_AGG": features_multi.get("aggregate", {}),
        "CANDIDATES": candidates,
        "CANDIDATE_RULES": rules_text
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

app = FastAPI(title="saasdj-backend (v1.1)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v1.1"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Recebe .mp3/.wav, extrai features multi-janela e classifica.
    Retorno: {"bpm": <int|None>, "subgenero": <str>, "confidence": <int>}
    """
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 1) Features multi-janela + aggregate
        feats_multi = extract_features_multi(data)
        agg = feats_multi["aggregate"]

        # 2) Candidatos por BPM (usa BPM do aggregate)
        bpm_val = agg.get("bpm")
        cands = candidates_by_bpm(bpm_val)

        # 3) Chamar GPT
        try:
            content = call_gpt(feats_multi, cands)
        except Exception as e:
            # Fallback heurístico melhorado
            fb_sub, fb_conf = backend_fallback_best_candidate(agg, cands)
            bpm_int = int(round(bpm_val)) if bpm_val is not None else None
            return JSONResponse(status_code=502, content={
                "bpm": bpm_int,
                "subgenero": fb_sub if fb_sub in SOFT_RULES else "Subgênero Não Identificado",
                "confidence": fb_conf if fb_sub in SOFT_RULES else 0,
                "error": str(e),
            })

        # 4) Parse "Subgênero:" / "Confiança:"
        sub = "Subgênero Não Identificado"; conf = 0
        for line in content.splitlines():
            L = line.strip().lower()
            if L.startswith("subgênero:") or L.startswith("subgenero:") or L.startswith("subgénero:"):
                sub = line.split(":", 1)[1].strip()
            elif L.startswith("confiança:") or L.startswith("confianca:"):
                try:
                    conf = int("".join(ch for ch in line.split(":", 1)[1] if ch.isdigit()))
                except Exception:
                    conf = 0

        # 5) Sanitização universo permitido
        if sub != "Subgênero Não Identificado" and sub not in SOFT_RULES:
            sub, conf = "Subgênero Não Identificado", 0

        # 6) Fallback se LLM não identificar
        if sub == "Subgênero Não Identificado":
            fb_sub, fb_conf = backend_fallback_best_candidate(agg, cands)
            if fb_sub != "Subgênero Não Identificado":
                sub, conf = fb_sub, max(conf, fb_conf)

        bpm_out = int(round(bpm_val)) if bpm_val is not None else None
        return {"bpm": bpm_out, "subgenero": sub, "confidence": conf}

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "confidence": 0,
            "error": f"processing failed: {e.__class__.__name__}: {e}",
        })
