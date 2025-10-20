# novo_backend_saasdj.py

"""
Este é um backend FastAPI completo para classificação de subgêneros eletrônicos usando:
- Extração de features com librosa/pydub
- Interpretação estilística via GPT (prompt interno)
- Fallback heurístico se a OpenAI falhar

Para rodar:
- Exporte a variável de ambiente OPENAI_API_KEY
- Rode com: `uvicorn novo_backend_saasdj:app --reload`
"""

import os
import io
import json
import numpy as np
import requests
import librosa
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -----------------------------
# Configurações
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

ALLOWED_EXTENSIONS = (".mp3", ".wav")

# -----------------------------
# Extração de features via librosa
# -----------------------------
def extract_audio_features(audio_bytes: bytes, start_time: float = 0.0, duration: float = 30.0):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    y, sr = librosa.load(wav_io, sr=None, offset=start_time, duration=duration)

    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    if bpm < 90:
        bpm *= 2

    spec = np.abs(librosa.stft(y))
    centroid = librosa.feature.spectral_centroid(S=spec).mean()
    bandwidth = librosa.feature.spectral_bandwidth(S=spec).mean()
    zcr = librosa.zero_crossings(y, pad=False).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()

    return {
        "bpm": round(float(bpm), 1),
        "spectral_centroid": round(float(centroid), 1),
        "spectral_bandwidth": round(float(bandwidth), 1),
        "zero_crossing_rate": round(float(zcr), 6),
        "mfcc": [round(x, 3) for x in mfccs[:5]]  # resumo compacto
    }

# -----------------------------
# Prompt e chamada ao GPT
# -----------------------------
GPT_PROMPT = """
Você é um especialista em música eletrônica.

Seu trabalho é analisar as seguintes features extraídas de uma faixa e determinar o subgênero apropriado a partir da lista abaixo:

Lista oficial de subgêneros (use apenas um):
House: Deep House, Tech House, Minimal Bass (Tech House), Progressive House, Bass House, Funky / Soulful House, Brazilian Bass, Future House, Afro House
Techno: Detroit Techno, Acid Techno, Industrial Techno, Peak Time Techno, Hard Techno, Melodic Techno, High-Tech Minimal
Trance: Uplifting Trance, Progressive Trance, Psytrance, Dark Psytrance
EDM: Big Room, Progressive EDM
Hard Dance: Hardstyle, Rawstyle, Gabber Hardcore, UK/Happy Hardcore, Jumpstyle
Bass Music: Dubstep, Drum & Bass, Liquid DnB, Neurofunk
Outros: Indie Dance

Baseie-se nas seguintes features:
- BPM
- Centroid e Bandwidth (espectro)
- Zero Crossing Rate (vocais / textura)
- MFCCs (perfil tonal da faixa)

Explique sua escolha em 1 ou 2 frases e retorne no seguinte formato:

Subgênero identificado: <um dos nomes da lista>
Explicação da escolha: <explicação curta>

Se não houver correspondência clara, retorne:
Subgênero identificado: Gênero não classificado
Explicação da escolha: <por que a faixa não encaixa>
"""

def call_openai_classifier(features: dict) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": GPT_PROMPT},
        {"role": "user", "content": json.dumps(features, ensure_ascii=False)}
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0.2}

    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Subgenre Classifier (GPT Logic)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "service": "subgenre-gpt-classifier"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Formato não suportado (somente .mp3 ou .wav)")

        audio_data = await file.read()
        features = extract_audio_features(audio_data)
        response = call_openai_classifier(features)

        sub, expl = "Gênero não classificado", ""
        for line in response.splitlines():
            if line.lower().startswith("subgênero identificado"):
                sub = line.split(":", 1)[1].strip()
            elif line.lower().startswith("explica"):
                expl = line.split(":", 1)[1].strip()

        return {"bpm": features["bpm"], "subgenero": sub, "explicacao": expl}

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bpm": None,
            "subgenero": "Erro",
            "explicacao": f"Erro na classificação: {type(e).__name__}: {e}"
        })
