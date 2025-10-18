import os, io, json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa, soundfile as sf
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

SUBGENRES = [
    "Deep House","Tech House","Minimal Bass (Tech House)","Progressive House","Bass House",
    "Funky / Soulful House","Brazilian Bass","Future House","Afro House","Indie Dance",
    "Detroit Techno","Acid Techno","Industrial Techno","Peak Time Techno","Hard Techno","Melodic Techno","High-Tech Minimal",
    "Uplifting Trance","Progressive Trance","Psytrance","Dark Psytrance",
    "Big Room","Progressive EDM",
    "Hardstyle","Rawstyle","Gabber Hardcore","UK/Happy Hardcore","Jumpstyle",
    "Dubstep","Drum & Bass","Liquid DnB","Neurofunk"
]

PROMPT = f"""
Você é um especialista em música eletrônica. Analise as features e classifique a faixa em um único subgênero da lista abaixo.
Retorne:
Subgênero: <um da lista>
Explicação: <1–2 frases baseadas em BPM, timbres e estrutura>

Se não houver correspondência, retorne:
Subgênero: Uncategorized Genre
Explicação: Não encontrei padrões compatíveis.

Subgêneros: {", ".join(SUBGENRES)}
"""

app = FastAPI(title="saasdj-backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def load_audio(file_bytes, sr=22050, max_seconds=90):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True, duration=max_seconds)
        if y is None or len(y) == 0:
            raise ValueError("Audio vazio")
        return y, sr
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        if len(audio) > max_seconds*1000:
            audio = audio[:max_seconds*1000]
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr

def extract_features(y, sr):
    tempo = librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=None)
    bpm = float(np.median(tempo)) if tempo is not None and len(tempo) > 0 else None
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1/sr)
    low = yf[(xf>=20) & (xf<120)].sum()
    mid = yf[(xf>=120) & (xf<2000)].sum()
    high = yf[(xf>=2000)].sum()
    return {"bpm": bpm, "energy_low": float(low), "energy_mid": float(mid), "energy_high": float(high)}

def call_gpt(features):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Features: {json.dumps(features)}"}
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0.1}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    return r.json()["choices"][0]["message"]["content"]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav")):
        raise HTTPException(400, "Envie arquivos .mp3 ou .wav")
    data = await file.read()
    y, sr = load_audio(data)
    feats = extract_features(y, sr)
    result = call_gpt(feats)
    return {"features": feats, "result": result}
