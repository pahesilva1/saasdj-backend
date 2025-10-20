# -*- coding: utf-8 -*-
"""
SaaSDJ Backend v1.3 – Musical (inspirado no analisador assertivo)

Mudanças principais:
✅ Janela fixa de 30s no centro da música (drop principal)
✅ Features intuitivas e musicais:
   - BPM
   - Low / Mid / High energy %
   - Onset Strength (força rítmica média)
   - HP Ratio (harmônico/percussivo)
✅ Prompt orientado a padrões musicais (interpretação semântica, não técnica)
✅ Sem MFCC, centroid, bandwidth ou zcr (reduz ruído e melhora entendimento)
✅ Leve, rápido e de alta assertividade
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
# FUNÇÕES DE ÁUDIO
# ==============================

def load_audio_center_segment(file_bytes: bytes, sr: int = 22050, segment_duration: float = 30.0):
    """Carrega áudio, converte para mono, pega o trecho central de 30s."""
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max

    total_duration = len(samples) / sr
    start = max(0, int((total_duration / 2 - segment_duration / 2) * sr))
    end = min(len(samples), int((total_duration / 2 + segment_duration / 2) * sr))
    segment = samples[start:end]

    return segment, sr, total_duration


def extract_features(segment: np.ndarray, sr: int) -> dict:
    """
    Extrai features musicais de um trecho mono:
      - BPM (estimativa robusta com 2 métodos + correção half-time)
      - Distribuição espectral (% Low/Mid/High)
      - HP Ratio (Harmônico/ Percussivo) com fallback seguro
      - Onset Strength (força rítmica média normalizada)
    Retorna tipos nativos do Python.
    """
    # ---------------------------
    # BPM (duas estimativas + escolha por bandas comuns)
    # ---------------------------
    try:
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    except Exception:
        onset_env = np.array([], dtype=np.float32)

    # Estimativa A: média do vetor de tempos (se existir)
    bpm_a = None
    if onset_env.size:
        try:
            tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
            if tempos is not None and len(tempos) > 0:
                bpm_a = float(np.mean(tempos))
        except Exception:
            bpm_a = None

    # Estimativa B: beat_track direto
    bpm_b = None
    try:
        tempo_bt, _ = librosa.beat.beat_track(y=segment, sr=sr)
        if tempo_bt and float(tempo_bt) > 0:
            bpm_b = float(tempo_bt)
    except Exception:
        bpm_b = None

    # Corrige half-time e coleciona candidatas
    candidates_bpm: list[float] = []
    for v in (bpm_a, bpm_b):
        if v is None:
            continue
        vv = v * 2.0 if v < 90.0 else v
        candidates_bpm.append(vv)

    # Escolhe a mais próxima de bandas frequentes de música eletrônica
    if candidates_bpm:
        bands = [(124, 130), (130, 138), (136, 142), (150, 160), (170, 178)]
        def dist_to_bands(x: float) -> float:
            best = float("inf")
            for a, b in bands:
                if a <= x <= b:
                    return 0.0
                best = min(best, abs(x - a), abs(x - b))
            return best
        bpm_val = min(candidates_bpm, key=dist_to_bands)
    else:
        bpm_val = 128.0  # valor seguro padrão

    # ---------------------------
    # FFT: distribuição de energia por bandas
    # ---------------------------
    try:
        spectrum = np.abs(rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)

        low_band = (freqs >= 20) & (freqs < 250)
        mid_band = (freqs >= 250) & (freqs < 4000)
        high_band = (freqs >= 4000) & (freqs <= 20000)

        low_energy = float(np.sum(spectrum[low_band]))
        mid_energy = float(np.sum(spectrum[mid_band]))
        high_energy = float(np.sum(spectrum[high_band]))

        total_energy = max(low_energy + mid_energy + high_energy, 1e-9)
        low_pct = float(round((low_energy / total_energy) * 100.0, 2))
        mid_pct = float(round((mid_energy / total_energy) * 100.0, 2))
        high_pct = float(round((high_energy / total_energy) * 100.0, 2))
    except Exception:
        low_pct, mid_pct, high_pct = 33.33, 33.33, 33.34  # fallback neutro

    # ---------------------------
    # HPSS: razão Harmônico/ Percussivo (com fallback)
    # ---------------------------
    try:
        harmonic, percussive = librosa.effects.hpss(segment)
        h_mean = float(np.mean(np.abs(harmonic))) if harmonic.size else 0.0
        p_mean = float(np.mean(np.abs(percussive))) if percussive.size else 1e-8
        hp_ratio = float(round((h_mean / p_mean) if p_mean > 0 else 1.0, 2))
    except Exception:
        hp_ratio = 1.0  # equilíbrio como fallback

    # ---------------------------
    # Onset Strength normalizado
    # ---------------------------
    if onset_env.size:
        denom = float(np.max(onset_env)) if float(np.max(onset_env)) > 0 else 1.0
        onset_strength = float(round(float(np.mean(onset_env)) / denom, 3))
    else:
        onset_strength = 0.0

    # ---------------------------
    # Saída (tipos nativos)
    # ---------------------------
    features = {
        "bpm": int(round(bpm_val)),
        "low_pct": float(low_pct),
        "mid_pct": float(mid_pct),
        "high_pct": float(high_pct),
        "hp_ratio": float(hp_ratio),
        "onset_strength": float(onset_strength),
    }
    return features


def candidates_by_bpm(bpm: float) -> list[str]:
    if bpm is None:
        return SUBGENRES[:]  # tudo, se não deu pra estimar

    b = bpm
    cands = []

    def add(xs): 
        for x in xs:
            if x not in cands: cands.append(x)

    # House / Indie Dance (118–126)
    if 116 <= b <= 127:
        add(["Deep House","Funky / Soulful House","Indie Dance","Progressive House",
             "Tech House","Minimal Bass (Tech House)","Bass House","Brazilian Bass",
             "Future House","Afro House","Melodic Techno","High-Tech Minimal","Detroit Techno"])

    # Techno/Peak (126–136)
    if 124 <= b <= 138:
        add(["Tech House","Peak Time Techno","High-Tech Minimal","Melodic Techno","Industrial Techno",
             "Acid Techno","Detroit Techno","Progressive House","Progressive EDM","Big Room"])

    # Trance (134–142)
    if 132 <= b <= 144:
        add(["Progressive Trance","Uplifting Trance","Psytrance","Dark Psytrance",
             "Melodic Techno","Peak Time Techno"])

    # Hard Techno/Hard Dance (145–165)
    if 142 <= b <= 166:
        add(["Hard Techno","Hardstyle","Rawstyle","Jumpstyle","UK/Happy Hardcore","Gabber Hardcore"])

    # Dubstep (half-time ~140)
    if 134 <= b <= 146:
        add(["Dubstep"])

    # Drum & Bass (170–180)
    if 166 <= b <= 186:
        add(["Drum & Bass","Liquid DnB","Neurofunk"])

    # se por acaso alguma janela caiu fora, ainda garante algo
    return cands or SUBGENRES[:]

# ==============================
# CHAMADA GPT
# ==============================

PROMPT = """
Você é um especialista em música eletrônica.
Receberá FEATURES de uma faixa e uma lista CANDIDATES de subgêneros plausíveis (filtrados por BPM).
Sua tarefa é escolher EXATAMENTE **um** subgênero dentre CANDIDATES. NÃO use rótulos fora de CANDIDATES.

Interprete as FEATURES pelos intervalos típicos:
- BPM (faixas aproximadas): 118–126 (House/Indie), 124–130 (Tech House/Prog House/Melodic Techno),
  128–136 (Techno pico), 134–142 (Trance), 145–165 (Hard Techno/Hard Dance), 170–180 (Drum & Bass).
- Low/Mid/High (% energia):
  • Low alto (45–60%) → kick/bass fortes (Techno, Tech House)
  • Mid alto (35–50%) → melódico/progressivo (Melodic Techno, Progressive, Trance)
  • High alto (25–40%) → brilho/hi-hats/impacto (EDM, Peak Time/Big Room)
- HP Ratio (harmônico/percussivo):
  • <0.9 → percussivo/seco (Techno/Hard)
  • 0.9–1.2 → equilibrado (Tech House/Prog House/Peak Time)
  • >1.2 → melódico/atmosférico (Melodic Techno/Prog/Uplifting)
- Onset strength:
  • 0.2–0.5 → grooves suaves (Deep/Indie)
  • 0.5–0.7 → fluido/progressivo (Prog/Melodic)
  • 0.7–1.0 → batida seca/direta (Tech House/Peak/Hard)

Responda em UMA linha, exatamente:
Subgênero: <um valor presente em CANDIDATES>
"""


def call_gpt(features: dict, candidates: list[str]) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    # Tipos nativos
    features = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in features.items()}

    payload = {
        "FEATURES": features,
        "CANDIDATES": candidates,
    }

    user_message = (
        "Classifique usando APENAS um rótulo presente em CANDIDATES, com base em FEATURES.\n"
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

app = FastAPI(title="SaaSDJ Backend v1.3 – Musical")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "saasdj-backend", "version": "v1.3"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classifica uma faixa em um subgênero eletrônico (base musical)."""
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(status_code=400, detail="Envie arquivos .mp3 ou .wav")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        segment, sr, dur = load_audio_center_segment(data)
        feats = extract_features(segment, sr)

        try:
            response = call_gpt(feats)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Erro na API do GPT: {e}")

        sub = "Subgênero Não Identificado"
        explic = ""
        for line in response.splitlines():
            l = line.strip().lower()
            if l.startswith("subgênero:") or l.startswith("subgenero:"):
                sub = line.split(":", 1)[1].strip()
            elif l.startswith("explicação:") or l.startswith("explicacao:"):
                explic = line.split(":", 1)[1].strip()

        if sub not in SUBGENRES:
            sub = "Subgênero Não Identificado"

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
