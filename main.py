import os
import numpy as np
import librosa
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from typing import Dict
import openai

# Inicializa o app FastAPI
app = FastAPI()

# Libera CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chave da API da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")

# Lista oficial de subgêneros permitidos
SUBGENRES = [
    "Deep House", "Tech House", "Minimal Bass (Tech House)", "Progressive House", "Bass House",
    "Funky / Soulful House", "Brazilian Bass", "Future House", "Afro House", "Indie Dance",
    "Detroit Techno", "Acid Techno", "Industrial Techno", "Peak Time Techno", "Hard Techno",
    "Melodic Techno", "High-Tech Minimal", "Uplifting Trance", "Progressive Trance", "Psytrance",
    "Dark Psytrance", "Big Room", "Progressive EDM"
]

# Extração de features
def extract_features(file_path: str) -> Dict:
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    sr = 22050
    total_samples = len(samples)
    center = total_samples // 2
    half_window = 15 * sr
    trimmed = samples[center - half_window:center + half_window]

    onset_env = librosa.onset.onset_strength(y=trimmed, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    if bpm < 90:
        bpm *= 2

    fft = np.abs(rfft(trimmed))
    freqs = rfftfreq(len(trimmed), 1 / sr)

    low = np.sum(fft[(freqs >= 20) & (freqs < 250)])
    mid = np.sum(fft[(freqs >= 250) & (freqs < 4000)])
    high = np.sum(fft[(freqs >= 4000) & (freqs <= 20000)])
    total = low + mid + high

    return {
        "bpm": round(bpm),
        "onset_strength": float(np.percentile(onset_env, 85)),
        "low": float(low / total),
        "mid": float(mid / total),
        "high": float(high / total),
    }

# Geração do prompt para o GPT
def build_prompt(features: Dict) -> str:
    return f"""
Você é um especialista em música eletrônica. Sua tarefa é identificar o subgênero de uma faixa com base nas seguintes características extraídas do áudio:

- BPM: {features['bpm']}
- Proporção de frequências:
  - Baixas (20–250 Hz): {features['low']:.2f}
  - Médias (250–4000 Hz): {features['mid']:.2f}
  - Altas (4000–20000 Hz): {features['high']:.2f}
- Força de transientes (onset strength, percentil 85%): {features['onset_strength']:.2f}

Considere apenas os seguintes subgêneros como possíveis:
{", ".join(SUBGENRES)}

Retorne apenas o nome do subgênero mais provável (somente um), seguido por uma breve explicação (1–2 frases) com base nos dados acima. Nunca invente subgêneros fora da lista.
"""

# Classificação com GPT
def classify_genre(features: Dict) -> str:
    prompt = build_prompt(features)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Você é um especialista em subgêneros de música eletrônica."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Endpoint da API
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        features = extract_features(temp_path)
        sub = classify_genre(features)

        return {
            "bpm": features["bpm"],
            "subgenero": sub
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

