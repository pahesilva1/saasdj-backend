# -*- coding: utf-8 -*-
"""
SaaSDJ Backend v1.6 – 3 janelas + BPM máximo + heurística de pré-ranque

Mudanças principais:
- 3 janelas de 30s (início ~1:00, centro, fim)
- BPM final = maior BPM entre as janelas (mitiga breakdowns)
- Janela de análise = maior "punch de pista" (onset + percussivo)
- Heurística de pré-ranque afinada (Tech/Minimal vs EDM vs Melodic vs Hard)
"""

import os
import io
import json
import numpy as np
import librosa
import requests
from scipy.fft import rfft
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ==============================
# CONFIGURAÇÕES
# ==============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

SUBGENRES = [
    # House
    "Deep House", "Tech House", "Minimal Bass (Tech House)", "Progressive House",
    "Bass House", "Funky / Soulful House", "Brazilian Bass", "Future House",
    "Afro House", "Indie Dance",
    # Techno
    "Detroit Techno", "Acid Techno", "Industrial Techno", "Peak Time Techno",
    "Hard Techno", "Melodic Techno", "High-Tech Minimal",
    # Trance
    "Uplifting Trance", "Progressive Trance", "Psytrance", "Dark Psytrance",
    # EDM (Festival)
    "Big Room", "Progressive EDM",
    # Hard Dance
    "Hardstyle", "Rawstyle", "Gabber Hardcore", "UK/Happy Hardcore", "Jumpstyle",
    # Bass Music
    "Dubstep", "Drum & Bass", "Liquid DnB", "Neurofunk",
]


# ==============================
# CARREGAMENTO E JANELAS
# ==============================

def load_audio(file_bytes: bytes, sr: int = 22050):
    """Carrega áudio com pydub, mono + sr fixo, retorna samples float32 [-1,1] e sr."""
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples, sr


def slice_windows(samples: np.ndarray, sr: int, seg_dur: float = 30.0):
    """Gera 3 janelas de 30s: início (~0:45–1:15), centro e fim (últimos 30s)."""
    n = len(samples)
    total_sec = n / sr
    L = int(seg_dur * sr)

    # Janela A: início (~0:45)
    a_start_sec = 45.0
    if total_sec < 75.0:
        a_start = 0
    else:
        a_start = int(a_start_sec * sr)
    a_end = min(n, a_start + L)
    if a_end - a_start < L:
        a_start = max(0, a_end - L)
    win_a = samples[a_start:a_end]

    # Janela B: centro
    mid = n // 2
    b_start = max(0, mid - L // 2)
    b_end = min(n, b_start + L)
    if b_end - b_start < L:
        b_start = max(0, b_end - L)
    win_b = samples[b_start:b_end]

    # Janela C: fim (últimos 30s)
    c_end = n
    c_start = max(0, c_end - L)
    win_c = samples[c_start:c_end]

    return [win_a, win_b, win_c]


# ==============================
# FEATURES BÁSICAS
# ==============================

def _bpm_from_segment(segment: np.ndarray, sr: int) -> float | None:
    """Retorna bpm estimado (com correção half-time); None se não der."""
    if segment.size == 0:
        return None
    try:
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    except Exception:
        onset_env = np.array([], dtype=np.float32)

    bpms = []

    # A) média de tempos do onset_env
    if onset_env.size:
        try:
            tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
            if tempos is not None and len(tempos) > 0:
                bpms.append(float(np.mean(tempos)))
        except Exception:
            pass

    # B) beat_track direto
    try:
        tempo_bt, _ = librosa.beat.beat_track(y=segment, sr=sr)
        if tempo_bt and float(tempo_bt) > 0:
            bpms.append(float(tempo_bt))
    except Exception:
        pass

    if not bpms:
        return None

    # correção half-time
    bpms = [x * 2.0 if x < 90.0 else x for x in bpms]

    # snap leve a bandas comuns (para estabilidade)
    bands = [(124, 130), (130, 138), (136, 142), (140, 160), (170, 178)]
    def dist_to_bands(x: float) -> float:
        best = float("inf")
        for a, b in bands:
            if a <= x <= b:
                return 0.0
            best = min(best, abs(x - a), abs(x - b))
        return best

    return min(bpms, key=dist_to_bands)


def _onset_strength(segment: np.ndarray, sr: int) -> float:
    try:
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        if onset_env.size == 0:
            return 0.0
        denom = float(np.max(onset_env)) if float(np.max(onset_env)) > 0 else 1.0
        return float(round(float(np.mean(onset_env)) / denom, 3))
    except Exception:
        return 0.0


def _hp_ratio(segment: np.ndarray) -> float:
    try:
        H, P = librosa.effects.hpss(segment)
        h_mean = float(np.mean(np.abs(H))) if H.size else 0.0
        p_mean = float(np.mean(np.abs(P))) if P.size else 1e-8
        return float(round((h_mean / p_mean) if p_mean > 0 else 1.0, 2))
    except Exception:
        return 1.0


def _energy_bands(segment: np.ndarray, sr: int):
    try:
        spec = np.abs(rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
        low = float(np.sum(spec[(freqs >= 20) & (freqs < 250)]))
        mid = float(np.sum(spec[(freqs >= 250) & (freqs < 4000)]))
        high = float(np.sum(spec[(freqs >= 4000) & (freqs <= 20000)]))
        total = max(low + mid + high, 1e-9)
        return (
            float(round((low / total) * 100.0, 2)),
            float(round((mid / total) * 100.0, 2)),
            float(round((high / total) * 100.0, 2)),
        )
    except Exception:
        return 33.33, 33.33, 33.34


# ==============================
# ESCOLHA DA JANELA + BPM GLOBAL
# ==============================

def _window_punch_score(hp_ratio: float, onset: float) -> float:
    # favorece percussivo + batida “seca”: hp baixo + onset alto
    perc_gain = max(0.0, min(1.0, 1.5 - hp_ratio))  # hp=0.8→0.7 ; hp=1.2→0.3
    return 0.6 * onset + 0.4 * perc_gain


def extract_features_from_best_window(samples: np.ndarray, sr: int) -> dict:
    wins = slice_windows(samples, sr, seg_dur=30.0)

    # Calcula features por janela e guarda bpm de todas
    best = None
    best_score = -1.0
    bpm_candidates = []

    for w in wins:
        if w.size == 0:
            continue
        bpm = _bpm_from_segment(w, sr)
        if bpm is not None:
            # half-time já corrigido
            bpm_candidates.append(bpm)

        hp = _hp_ratio(w)
        on = _onset_strength(w, sr)
        low, mid, high = _energy_bands(w, sr)

        score = _window_punch_score(hp, on)
        if score > best_score:
            best_score = score
            best = {
                "low_pct": float(low),
                "mid_pct": float(mid),
                "high_pct": float(high),
                "hp_ratio": float(hp),
                "onset_strength": float(on),
            }

    # BPM global = MAIOR bpm entre janelas (minimiza subestimativa por breakdown)
    if bpm_candidates:
        bpm_final = int(round(max(bpm_candidates)))
    else:
        bpm_final = 128

    best["bpm"] = bpm_final
    return best


# ==============================
# CANDIDATES + PRÉ-RANQUE
# ==============================

def candidates_by_bpm(bpm: float | int | None) -> list[str]:
    """Retorna subgêneros plausíveis dado o BPM estimado (com janelas sobrepostas)."""
    if bpm is None:
        return SUBGENRES[:]

    b = float(bpm)
    cands: list[str] = []

    def add(xs):
        for x in xs:
            if x not in cands:
                cands.append(x)

    # House / Indie (118–126)
    if 116 <= b <= 127:
        add([
            "Tech House","Minimal Bass (Tech House)","Bass House",
            "Progressive House","Brazilian Bass","Future House",
            "Deep House","Funky / Soulful House","Afro House","Indie Dance",
            "Melodic Techno","High-Tech Minimal","Detroit Techno"
        ])

    # Techno / Peak (126–136)
    if 124 <= b <= 138:
        add([
            "Tech House","Peak Time Techno","High-Tech Minimal","Melodic Techno","Industrial Techno",
            "Acid Techno","Detroit Techno","Progressive House","Progressive EDM","Big Room"
        ])

    # Trance (132–144)
    if 132 <= b <= 144:
        add([
            "Progressive Trance","Uplifting Trance","Psytrance","Dark Psytrance",
            "Melodic Techno","Peak Time Techno"
        ])

    # Hard Techno / Hard Dance (140–165)  ← começa em 140 para não perder casos na borda
    if 140 <= b <= 165:
        add([
            "Hard Techno","Hardstyle","Rawstyle","Jumpstyle","UK/Happy Hardcore","Gabber Hardcore"
        ])

    # Dubstep (half-time ~140)
    if 136 <= b <= 146:
        add(["Dubstep"])

    # Drum & Bass (170–180)
    if 166 <= b <= 186:
        add(["Drum & Bass","Liquid DnB","Neurofunk"])

    return cands or SUBGENRES[:]


def prerank_candidates(cands: list[str], feats: dict) -> list[str]:
    """Reordena CANDIDATES com heurística simples baseada nas features."""
    if not cands:
        return cands

    bpm = feats["bpm"]
    low = feats["low_pct"]; mid = feats["mid_pct"]; high = feats["high_pct"]
    hp = feats["hp_ratio"]; on = feats["onset_strength"]

    priority = []

    # A) Tech House / Minimal / Bass House (kick/sub forte, pouco brilho)
    if low >= 42 and hp <= 1.10 and high <= 30 and on >= 0.60:
        priority += ["Tech House", "Minimal Bass (Tech House)", "Bass House", "Brazilian Bass"]

    # B) EDM / Peak / Big Room (brilho/impacto com equilíbrio)
    if high >= 24 and 0.9 <= hp <= 1.25 and on >= 0.65:
        priority += ["Progressive EDM", "Big Room", "Peak Time Techno", "Future House"]

    # C) Melódico/Trance (só quando coerente; evita puxar tudo pra Melodic)
    if hp >= 1.15 and mid >= 36 and on <= 0.80:
        priority += ["Melodic Techno", "Progressive Trance", "Uplifting Trance"]

    # D) Hard Techno (bordas)
    if bpm >= 140 and hp <= 1.0 and low >= 40:
        priority = ["Hard Techno"] + priority

    # Anti-viés Indie Dance (só se combinar muito)
    if not (bpm <= 125 and on <= 0.60 and hp >= 1.10):
        if "Indie Dance" in cands:
            cands = [x for x in cands if x != "Indie Dance"] + ["Indie Dance"]

    # Dedup mantendo ordem: priority primeiro, depois o resto
    seen = set()
    ordered = []
    for x in priority + cands:
        if x in cands and x not in seen:
            ordered.append(x); seen.add(x)
    return ordered


# ==============================
# PROMPT + GPT
# ==============================

PROMPT = """
Você é um especialista em música eletrônica.
Receberá FEATURES de uma faixa e uma lista CANDIDATES de subgêneros plausíveis (filtrados por BPM).
Sua tarefa é escolher EXATAMENTE um subgênero dentre CANDIDATES. NÃO use rótulos fora de CANDIDATES.

Interprete as FEATURES pelos intervalos típicos:
- BPM (faixas aproximadas): 118–126 (House/Indie), 124–130 (Tech House/Prog House/Melodic Techno),
  128–136 (Techno pico), 132–144 (Trance), 140–165 (Hard Techno/Hard Dance), 170–180 (Drum & Bass).
- Low/Mid/High (% energia):
  • Low alto (45–60%) → kick/bass fortes (Techno, Tech House)
  • Mid alto (35–50%) → melódico/progressivo (Melodic Techno, Progressive, Trance)
  • High alto (24–40%) → brilho/hi-hats/impacto (EDM, Peak Time/Big Room)
- HP Ratio (harmônico/percussivo):
  • <0.9 → percussivo/seco (Techno/Hard)
  • 0.9–1.2 → equilibrado (Tech House/Prog House/Peak/EDM)
  • >1.2 → melódico/atmosférico (Melodic/Prog/Uplifting)
- Onset strength:
  • 0.2–0.5 → grooves suaves (Deep/Indie)
  • 0.5–0.7 → fluido/progressivo (Prog/Melodic)
  • 0.7–1.0 → batida seca/direta (Tech/Peak/Hard)

Regras adicionais:
- Não escolha "Indie Dance" a menos que BPM ≤ 125, onset_strength ≤ 0.60 e hp_ratio ≥ 1.10.
- Se houver conflito entre rótulos melódicos (Melodic/Trance) e rótulos percussivos (Tech/Hard/EDM),
  use hp_ratio como desempate: hp_ratio alto favorece Melodic/Trance; hp_ratio baixo favorece Tech/Hard/EDM.

Responda em UMA linha, exatamente:
Subgênero: <um valor presente em CANDIDATES>
""".strip()


def call_gpt(features: dict, candidates: list[str]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    # Tipos nativos
    features = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in features.items()}
    payload = {"FEATURES": features, "CANDIDATES": candidates}
    user_message = (
        "Classifique usando APENAS um rótulo presente em CANDIDATES, com base em FEATURES.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,
    }

    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"].strip()


# ==============================
# FASTAPI
# ==============================

app = FastAPI(title="SaaSDJ Backend v1.6 – Musical")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v1.6"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classifica faixa em um subgênero eletrônico (resposta: arquivo, bpm, subgenero)."""
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(status_code=400, detail="Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        samples, sr = load_audio(data)
        feats = extract_features_from_best_window(samples, sr)

        # Candidatos por BPM + pré-ranque
        cands = candidates_by_bpm(feats["bpm"])
        cands = prerank_candidates(cands, feats)

        try:
            response = call_gpt(feats, cands)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Erro na API do GPT: {e}")

        # Parse: "Subgênero: X"
        sub = None
        for line in response.splitlines():
            l = line.strip().lower()
            if l.startswith("subgênero:") or l.startswith("subgenero:"):
                sub = line.split(":", 1)[1].strip()
                break

        # Fallback: se não vier ou vier fora do set, pega 1º candidato pós-pré-ranque
        if not sub or sub not in SUBGENRES or sub not in cands:
            sub = cands[0] if cands else "Subgênero Não Identificado"

        return JSONResponse({
            "arquivo": file.filename,
            "bpm": feats["bpm"],
            "subgenero": sub,
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "erro": f"Falha inesperada: {e.__class__.__name__}: {e}"
        })
