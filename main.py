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
    """Extrai BPM, distribuição espectral, HP ratio e força rítmica."""
    # BPM
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    bpm = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
    if bpm < 90:
        bpm *= 2  # corrige half-time

    # FFT
    spectrum = np.abs(rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1 / sr)

    low_band = (freqs >= 20) & (freqs < 250)
    mid_band = (freqs >= 250) & (freqs < 4000)
    high_band = (freqs >= 4000) & (freqs <= 20000)

    low_energy = np.sum(spectrum[low_band])
    mid_energy = np.sum(spectrum[mid_band])
    high_energy = np.sum(spectrum[high_band])
    total_energy = max(low_energy + mid_energy + high_energy, 1e-9)

    low_pct = round((low_energy / total_energy) * 100, 2)
    mid_pct = round((mid_energy / total_energy) * 100, 2)
    high_pct = round((high_energy / total_energy) * 100, 2)

    # HPSS
    harmonic, percussive = librosa.effects.hpss(segment)
    hp_ratio = np.mean(np.abs(harmonic)) / np.mean(np.abs(percussive))

    # Força rítmica média
    onset_strength = float(np.mean(onset_env) / np.max(onset_env))

    features = {
        "bpm": float(round(bpm)),
        "low_pct": float(low_pct),
        "mid_pct": float(mid_pct),
        "high_pct": float(high_pct),
        "hp_ratio": float(round(hp_ratio, 2)),
        "onset_strength": float(round(onset_strength, 3)),
    }
    return features


# ==============================
# CHAMADA GPT
# ==============================

PROMPT = """
Você é um especialista em música eletrônica. 
Receberá dados técnicos sobre uma faixa (BPM, proporção de energia em graves/médios/agudos, razão harmônica-percussiva e força rítmica).
Sua tarefa é classificar a faixa em **um único subgênero** da lista abaixo.

A resposta deve ser **apenas duas linhas**:
Subgênero: <nome exato>
Explicação: <1 frase curta explicando com base nos dados>

Subgêneros possíveis:
Deep House, Tech House, Minimal Bass (Tech House), Progressive House, Bass House,
Funky / Soulful House, Brazilian Bass, Future House, Afro House, Indie Dance,
Detroit Techno, Acid Techno, Industrial Techno, Peak Time Techno, Hard Techno,
Melodic Techno, High-Tech Minimal, Uplifting Trance, Progressive Trance, Psytrance,
Dark Psytrance, Big Room, Progressive EDM, Hardstyle, Rawstyle, Gabber Hardcore,
UK/Happy Hardcore, Jumpstyle, Dubstep, Drum & Bass, Liquid DnB, Neurofunk.

---

🎚️ Interprete as features com base em faixas musicais típicas (use **intervalos**, não valores fixos):

🔹 **BPM (faixa aproximada)**
- 118–125 → Deep / Funky / Soulful House, Indie Dance  
- 124–130 → Tech House, Progressive House, Melodic Techno  
- 128–136 → Techno (Peak Time, High-Tech, Melodic)  
- 134–142 → Trance (Progressive, Uplifting, Psy)  
- 145–160 → Hard Techno, Hard Dance  
- 170–180 → Drum & Bass, Liquid, Neurofunk  

🔹 **Low / Mid / High Energy (%)**
- Low alto (45–60%) → estilos centrados no kick/bassline (Techno, Tech House)  
- Mid alto (35–50%) → estilos melódicos e progressivos (Melodic Techno, Progressive House, Trance)  
- High alto (25–40%) → estilos energéticos, com hi-hats e brilho (EDM, Peak Time Techno, Big Room)  

🔹 **HP Ratio (Harmônico/Percussivo)**
- < 0.9 → percussivo e seco → Hard Techno, Peak Time, Tech House  
- 0.9–1.2 → equilibrado → Progressive House, Progressive EDM, Techno  
- > 1.2 → melódico e atmosférico → Melodic Techno, Progressive Trance, Uplifting Trance  

🔹 **Onset Strength (força rítmica)**
- 0.2–0.5 → grooves suaves ou deep → Deep House, Indie Dance  
- 0.5–0.7 → fluído → Progressive / Melodic estilos  
- 0.7–1.0 → batida seca, direta → Tech House, Peak Time, Hard Techno  

---

🎧 Exemplos referenciais (use como base de raciocínio, não como regra exata):

- **Melodic Techno:** BPM ~122–128, hp_ratio >1.2, mid_pct alto (melodias e atmosferas progressivas)
- **Hard Techno:** BPM 140–155, hp_ratio <0.9, low_pct alto (kick agressivo e pouca melodia)
- **Uplifting Trance:** BPM 136–140, hp_ratio >1.2, mid_pct alto, high_pct moderado (melódico e eufórico)
- **Tech House:** BPM 124–128, low_pct alto, hp_ratio 0.9–1.1, onset forte e groove seco
- **Progressive House:** BPM 122–128, mid_pct e hp_ratio equilibrados, flow contínuo e harmônico
- **Drum & Bass:** BPM 170–178, high_pct alto, hp_ratio baixo (ritmo frenético)

---

⚙️ Instruções finais:
1. Use **todos os dados juntos** — não confie apenas no BPM.  
2. Se os dados parecerem ambíguos, escolha o subgênero **mais provável musicalmente**.  
3. Se nada fizer sentido, retorne:
Subgênero: Uncategorized Genre
Explicação: Dados não coincidem claramente com nenhum subgênero.
"""


def call_gpt(features: dict) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    # 🔧 Corrige tipos NumPy → Python
    features = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in features.items()}
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(features, ensure_ascii=False)}
        ],
        "temperature": 0.3,
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
