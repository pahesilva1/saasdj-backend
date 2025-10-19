import os, io, json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa, soundfile as sf
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq
from fastapi.responses import JSONResponse
from typing import Dict, List, Tuple
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

# ---- SOFT RULES NUMÉRICAS (por subgênero) ----
# As bandas usam proporções (0–1) com base em low_pct, mid_pct, high_pct.
# hp_ratio = H/P (harmônico ÷ percussivo); onset_strength é a média do onset (faixas amplas).
SOFT_RULES: Dict[str, Dict] = {
    # -------- HOUSE --------
    "Deep House": {
        "bpm": (120, 124),
        "bands_pct": {"low": (0.20, 0.35), "mid": (0.40, 0.60), "high": (0.10, 0.25)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.20, 0.50),
        "signatures": "groove suave/profundo, acordes/pads, vocais quentes; menos foco em transientes"
    },
    "Tech House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.12, 0.28)},
        "hp_ratio": (0.75, 1.05),
        "onset_strength": (0.40, 0.65),
        "signatures": "kick/bass secos e funcionais, grooves repetitivos, poucos leads"
    },
    "Minimal Bass (Tech House)": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.45, 0.68), "mid": (0.16, 0.32), "high": (0.08, 0.22)},
        "hp_ratio": (0.70, 0.95),
        "onset_strength": (0.35, 0.60),
        "signatures": "sub muito forte e arranjo minimalista; foco no baixo e groove enxuto"
    },
    "Progressive House": {
        "bpm": (122, 128),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.58), "high": (0.15, 0.30)},
        "hp_ratio": (1.10, 1.70),
        "onset_strength": (0.25, 0.50),
        "signatures": "builds longos, atmosfera melódica, progressão constante/emotiva"
    },
    "Bass House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.45, 0.70), "mid": (0.20, 0.40), "high": (0.18, 0.35)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.45, 0.70),
        "signatures": "basslines agressivas/‘talking’, queda forte no drop (absorve Electro House)"
    },
    "Funky / Soulful House": {
        "bpm": (120, 125),
        "bands_pct": {"low": (0.25, 0.40), "mid": (0.40, 0.60), "high": (0.10, 0.25)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.20, 0.45),
        "signatures": "instrumentação orgânica, elementos soul/disco, presença de vocais"
    },
    "Brazilian Bass": {
        "bpm": (120, 126),
        "bands_pct": {"low": (0.50, 0.75), "mid": (0.18, 0.35), "high": (0.08, 0.22)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.35, 0.60),
        "signatures": "sub/slap marcante com groove pop-friendly, vocais ocasionais"
    },
    "Future House": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.35, 0.55), "mid": (0.22, 0.40), "high": (0.20, 0.38)},
        "hp_ratio": (0.95, 1.25),
        "onset_strength": (0.45, 0.70),
        "signatures": "timbres ‘future’/serrilhas, drops claros e brilhantes"
    },
    "Afro House": {
        "bpm": (118, 125),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.35, 0.55), "high": (0.12, 0.28)},
        "hp_ratio": (1.05, 1.60),
        "onset_strength": (0.35, 0.60),
        "signatures": "percussões afro, groove orgânico, vocais/texturas étnicas"
    },
    "Indie Dance": {
        "bpm": (110, 125),
        "bands_pct": {"low": (0.18, 0.35), "mid": (0.40, 0.60), "high": (0.12, 0.28)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.25, 0.50),
        "signatures": "vibe retrô/alternativa, synths vintage, menos ênfase em transientes"
    },

    # -------- TECHNO --------
    "Detroit Techno": {
        "bpm": (122, 130),
        "bands_pct": {"low": (0.28, 0.45), "mid": (0.30, 0.50), "high": (0.12, 0.28)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.40, 0.65),
        "signatures": "groove clássico/analógico, estética quente, linhas repetitivas"
    },
    "Acid Techno": {
        "bpm": (125, 135),
        "bands_pct": {"low": (0.28, 0.45), "mid": (0.35, 0.55), "high": (0.15, 0.32)},
        "hp_ratio": (0.95, 1.30),
        "onset_strength": (0.45, 0.70),
        "signatures": "timbre TB-303 em destaque (ressonante/squelch) como elemento central"
    },
    "Industrial Techno": {
        "bpm": (128, 140),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.20, 0.40)},
        "hp_ratio": (0.75, 1.05),
        "onset_strength": (0.60, 0.85),
        "signatures": "texturas industriais/ruidosas, sensação ‘fábrica’, percussão pesada"
    },
    "Peak Time Techno": {
        "bpm": (128, 132),
        "bands_pct": {"low": (0.32, 0.56), "mid": (0.24, 0.42), "high": (0.18, 0.35)},
        "hp_ratio": (0.85, 1.15),
        "onset_strength": (0.55, 0.80),
        "signatures": "4x4 direto para ápice; leads discretos; energia constante (ref. Victor Ruiz – All Night Long)"
    },
    "Hard Techno": {
        "bpm": (135, 150),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.70, 0.95),
        "onset_strength": (0.65, 0.90),
        "signatures": "agressivo e percussivo; kicks duros; pouca melodia"
    },
    "Melodic Techno": {
        "bpm": (122, 128),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.60), "high": (0.15, 0.32)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.30, 0.60),
        "signatures": "pads/leads emocionais e cinematográficos, progressão envolvente"
    },
    "High-Tech Minimal": {
        "bpm": (124, 130),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.22, 0.40), "high": (0.10, 0.25)},
        "hp_ratio": (0.85, 1.15),
        "onset_strength": (0.35, 0.60),
        "signatures": "minimalista, design sonoro detalhista; sub consistente, timbres enxutos"
    },

    # -------- TRANCE --------
    "Uplifting Trance": {
        "bpm": (134, 140),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.40, 0.60), "high": (0.15, 0.32)},
        "hp_ratio": (1.20, 2.00),
        "onset_strength": (0.35, 0.60),
        "signatures": "supersaws eufóricas, breakdowns grandes; absorve Vocal Trance eufórico"
    },
    "Progressive Trance": {
        "bpm": (132, 138),
        "bands_pct": {"low": (0.22, 0.40), "mid": (0.38, 0.60), "high": (0.15, 0.32)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.25, 0.50),
        "signatures": "atmosfera rolante; menos euforia que Uplifting; absorve Vocal Trance atmosférico"
    },
    "Psytrance": {
        "bpm": (138, 146),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.32, 0.52), "high": (0.15, 0.30)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.35, 0.60),
        "signatures": "rolling bass e FX psicodélicos; absorve Goa"
    },
    "Dark Psytrance": {
        "bpm": (145, 150),
        "bands_pct": {"low": (0.30, 0.50), "mid": (0.25, 0.45), "high": (0.20, 0.40)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.50, 0.75),
        "signatures": "mais escuro/denso; texturas e ruídos em destaque"
    },

    # -------- EDM (Festival) --------
    "Big Room": {
        "bpm": (126, 128),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.40), "high": (0.22, 0.42)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.60, 0.85),
        "signatures": "drops marcados; dinâmica alta; hi-hats/brilho dando impacto"
    },
    "Progressive EDM": {
        "bpm": (124, 128),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.40, 0.60), "high": (0.15, 0.30)},
        "hp_ratio": (1.10, 1.80),
        "onset_strength": (0.35, 0.60),
        "signatures": "melódico de festival, polido, progressivo/pop-friendly"
    },

    # -------- HARD DANCE --------
    "Hardstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.22, 0.42)},
        "hp_ratio": (0.80, 1.10),
        "onset_strength": (0.65, 0.90),
        "signatures": "kicks distorcidos/‘reverse bass’, intensidade alta"
    },
    "Rawstyle": {
        "bpm": (150, 160),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.22, 0.42), "high": (0.25, 0.45)},
        "hp_ratio": (0.75, 1.00),
        "onset_strength": (0.70, 0.95),
        "signatures": "mais agressivo/distorcido que Hardstyle; sound design extremo"
    },
    "Gabber Hardcore": {
        "bpm": (170, 190),
        "bands_pct": {"low": (0.35, 0.60), "mid": (0.20, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.70, 0.95),
        "onset_strength": (0.75, 0.98),
        "signatures": "muito agressivo, kicks ‘serra’, textura crua"
    },
    "UK/Happy Hardcore": {
        "bpm": (165, 180),
        "bands_pct": {"low": (0.28, 0.48), "mid": (0.35, 0.55), "high": (0.22, 0.42)},
        "hp_ratio": (1.00, 1.60),
        "onset_strength": (0.60, 0.85),
        "signatures": "rápido, eufórico/brilhante, melodias felizes"
    },
    "Jumpstyle": {
        "bpm": (140, 150),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.25, 0.45), "high": (0.15, 0.32)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.55, 0.80),
        "signatures": "padrões rítmicos ‘saltados’, batidas quadradas"
    },

    # -------- BASS MUSIC --------
    "Dubstep": {
        "bpm": (136, 146),
        "bands_pct": {"low": (0.45, 0.75), "mid": (0.20, 0.40), "high": (0.20, 0.40)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.60, 0.85),
        "signatures": "half-time ~140, wobble/sub pesado; absorve Riddim"
    },
    "Drum & Bass": {
        "bpm": (170, 180),
        "bands_pct": {"low": (0.30, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.90, 1.30),
        "onset_strength": (0.75, 0.98),
        "signatures": "breakbeat rápido, muitos ataques, energia constante"
    },
    "Liquid DnB": {
        "bpm": (170, 176),
        "bands_pct": {"low": (0.25, 0.45), "mid": (0.38, 0.58), "high": (0.15, 0.30)},
        "hp_ratio": (1.05, 1.60),
        "onset_strength": (0.45, 0.70),
        "signatures": "suave/atmosférico, pads/vocais; break mais limpo"
    },
    "Neurofunk": {
        "bpm": (172, 178),
        "bands_pct": {"low": (0.32, 0.55), "mid": (0.22, 0.40), "high": (0.25, 0.45)},
        "hp_ratio": (0.85, 1.20),
        "onset_strength": (0.65, 0.90),
        "signatures": "baixo serrado/complexo, agressivo e técnico"
    },
}


def candidates_by_bpm(bpm: float) -> List[str]:
    """
    Seleciona candidatos pela faixa de BPM de cada subgênero (SOFT_RULES).
    Bordas generosas (+/- 2 BPM) para não cortar casos reais.
    """
    if bpm is None:
        return list(SOFT_RULES.keys())

    margin = 4.0
    cands = []
    for name, meta in SOFT_RULES.items():
        lo, hi = meta["bpm"]
        if (bpm >= lo - margin) and (bpm <= hi + margin):
            cands.append(name)
    # fallback se vazio
    return cands or list(SOFT_RULES.keys())


def format_rules_for_candidates(cands: List[str]) -> str:
    """
    Constrói um texto compacto com faixas numéricas para os candidatos.
    """
    lines = []
    for name in cands:
        m = SOFT_RULES[name]
        lo_bpm, hi_bpm = m["bpm"]
        lp = m["bands_pct"]["low"]; mp = m["bands_pct"]["mid"]; hp = m["bands_pct"]["high"]
        hr_lo, hr_hi = m["hp_ratio"]
        lines.append(
            f"Nome: {name}\n"
            f"BPM: {lo_bpm}–{hi_bpm}\n"
            f"Bandas%: low={lp[0]:.2f}–{lp[1]:.2f} | mid={mp[0]:.2f}–{mp[1]:.2f} | high={hp[0]:.2f}–{hp[1]:.2f}\n"
            f"HP Ratio: {hr_lo:.2f}–{hr_hi:.2f}\n"
            f"Assinaturas: {m['signatures']}\n"
            f"---"
        )

    return "\n".join(lines)


PROMPT = """
Você é um especialista em música eletrônica. Classifique a faixa com base nas FEATURES abaixo.
As features vêm do trecho 60–120s (ou dos 60s finais se a faixa tiver menos de 2 minutos).

Use APENAS um subgênero dentre CANDIDATES.
Compare as FEATURES com as faixas numéricas em CANDIDATE_RULES (BPM, Bandas%, HP Ratio).
Calcule uma similaridade ponderada (não precisa mostrar): 
- BPM (peso 0.4): mais alto se o BPM cair dentro da faixa do candidato (ou perto do centro da faixa).
- Bandas% (peso 0.4): mais alto quanto mais low/mid/high_pct caírem nas faixas do candidato.
- HP Ratio (peso 0.2): mais alto se ficar dentro da faixa do candidato.

Escolha o candidato de MAIOR similaridade. 
Só use 'Subgênero Não Identificado' se a similaridade final for muito baixa (ex.: < 0.40).

Responda exatamente em DUAS linhas:
Subgênero: <um dos CANDIDATES ou 'Subgênero Não Identificado'>
Confiança: <número inteiro de 0 a 100>

Observações internas (não exponha):
- Absorções: Bass House ← Electro House; Uplifting/Progressive Trance ← Vocal Trance; Psytrance ← Goa; Dubstep ← Riddim.
- Não há necessidade de perfeição de faixa; priorize o candidato com melhor aderência geral às faixas numéricas.
"""


app = FastAPI(title="saasdj-backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def load_audio(file_bytes, sr=22050):
    """
    Regra de janela:
      - padrão: analisar 60–120s (60s)
      - se a faixa tiver < 120s: analisar os 60s finais
    Carrega mono, com resample para sr.
    """
    # 1) Medir duração com pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_sec = len(audio) / 1000.0
    except Exception:
        audio = None
        duration_sec = None

    # 2) Decidir offset/duration
    if duration_sec is None:
        # Sem duração: tenta direto 60–120s (melhor esforço)
        offset_seconds = 60.0
        duration_seconds = 60.0
    else:
        if duration_sec >= 120.0:
            offset_seconds = 60.0
            duration_seconds = 60.0
        else:
            # 60s antes do final (ou desde o começo se <60s)
            offset_seconds = max(0.0, duration_sec - 60.0)
            duration_seconds = min(60.0, duration_sec - offset_seconds)

    # 3) Carregar via librosa (principal) com offset/duration definidos
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True,
                             offset=offset_seconds, duration=duration_seconds)
        if y is None or len(y) == 0:
            raise ValueError("Audio vazio após leitura principal")
        return y, sr
    except Exception:
        # 4) Fallback: converter para wav com pydub e tentar de novo
        if audio is None:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        start_ms = int(offset_seconds * 1000.0)
        end_ms = int((offset_seconds + duration_seconds) * 1000.0)
        segment = audio[start_ms:end_ms]
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=sr, mono=True)
        return y, sr

def extract_features(y, sr):
    # ----- BPM (compat com librosa >=0.10) -----
    try:
        from librosa.feature.rhythm import tempo as tempo_fn
        tempo_vals = tempo_fn(y=y, sr=sr, max_tempo=200, aggregate=None)
    except Exception:
        tempo_vals = librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=None)

    bpm = float(np.median(tempo_vals)) if tempo_vals is not None and len(tempo_vals) > 0 else None
    # Correção apenas para half-time (NÃO dividir >180 para preservar Hard Dance)
    if bpm is not None and bpm < 90.0:
        bpm *= 2.0

    # ----- FFT bandas -----
    N = len(y)
    yf = np.abs(rfft(y))
    xf = rfftfreq(N, 1/sr)
    low = float(yf[(xf >= 20) & (xf < 120)].sum())
    mid = float(yf[(xf >= 120) & (xf < 2000)].sum())
    high = float(yf[(xf >= 2000)].sum())

    total = max(low + mid + high, 1e-9)
    low_pct = float(low / total)
    mid_pct = float(mid / total)
    high_pct = float(high / total)

    # ----- Harmônico/ Percussivo + Onset -----
    H, P = librosa.effects.hpss(y)
    hp_ratio = float((np.mean(np.abs(H)) + 1e-8) / (np.mean(np.abs(P)) + 1e-8))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = float(np.mean(onset_env)) if onset_env is not None and len(onset_env) else 0.0

    return {
        "bpm": round(bpm, 3) if bpm else None,
        "energy_low": low,
        "energy_mid": mid,
        "energy_high": high,
        "low_pct": round(low_pct, 6),
        "mid_pct": round(mid_pct, 6),
        "high_pct": round(high_pct, 6),
        "hp_ratio": round(hp_ratio, 6),
        "onset_strength": round(onset_strength, 6),
    }


def call_gpt(features: dict, candidates: List[str]):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    rules_text = format_rules_for_candidates(candidates)
    user_payload = {
        "FEATURES": features,
        "CANDIDATES": candidates,
        "CANDIDATE_RULES": rules_text
    }

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0}

    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=headers, json=data, timeout=60)

    try:
        body = r.json()
    except Exception:
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - resposta não-JSON: {r.text[:500]}")

    if r.status_code != 200:
        err = body.get("error", {})
        raise RuntimeError(f"OpenAI HTTP {r.status_code} - {err.get('type')} - {err.get('message')}")

    if "choices" not in body or not body["choices"]:
        raise RuntimeError(f"OpenAI resposta inesperada: {body}")

    content = body["choices"][0]["message"]["content"].strip()
    return content

def _score_in_range(val, rng):
    lo, hi = rng
    if val is None:
        return 0.0
    if val < lo:
        return max(0.0, 1.0 - (lo - val) / max(hi - lo, 1e-6))
    if val > hi:
        return max(0.0, 1.0 - (val - hi) / max(hi - lo, 1e-6))
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 + 1e-6
    return 1.0 + max(0.0, 0.2 * (1.0 - abs(val - mid) / half))

def backend_fallback_best_candidate(features: dict, candidates: list[str]) -> tuple[str, int]:
    bpm = features.get("bpm")
    lp = features.get("low_pct", 0.0)
    mp = features.get("mid_pct", 0.0)
    hp = features.get("high_pct", 0.0)
    hpr = features.get("hp_ratio", 0.0)

    best_name = "Subgênero Não Identificado"
    best_score = 0.0

    for name in candidates:
        rule = SOFT_RULES[name]
        s_bpm = _score_in_range(bpm, rule["bpm"])
        s_low = _score_in_range(lp, rule["bands_pct"]["low"])
        s_mid = _score_in_range(mp, rule["bands_pct"]["mid"])
        s_high = _score_in_range(hp, rule["bands_pct"]["high"])
        s_hp = _score_in_range(hpr, rule["hp_ratio"])

        bands_avg = (s_low + s_mid + s_high) / 3.0
        score = 0.4 * s_bpm + 0.4 * bands_avg + 0.2 * s_hp

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < 0.40:
        return "Subgênero Não Identificado", 0

    conf = int(min(95, max(50, 50 + (best_score - 0.40) * 100)))
    return best_name, conf



@app.get("/health")
def health():
    return {"ok": True}

from fastapi.responses import JSONResponse

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".mp3", ".wav")):
            raise HTTPException(400, "Envie arquivos .mp3 ou .wav")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Arquivo vazio")

        # 1) Carregar janela correta e extrair features
        y, sr = load_audio(data)
        feats = extract_features(y, sr)

        # 2) Selecionar candidatos por BPM
        bpm_val = feats.get("bpm")
        cands = candidates_by_bpm(bpm_val)

        # 3) Chamar GPT com candidatos e regras
        try:
            content = call_gpt(feats, cands)
        except Exception as e:
            # Falha na OpenAI — retornamos “Não Identificado”
            return JSONResponse(status_code=502, content={
                "bpm": bpm_val,
                "subgenero": "Subgênero Não Identificado",
                "confidence": 0,
                "error": str(e)
            })

        # 4) Parse da resposta "Subgênero: X" / "Confiança: NN"
        sub = "Subgênero Não Identificado"
        conf = 0
        for line in content.splitlines():
            line = line.strip()
            if line.lower().startswith("subgênero:"):
                sub = line.split(":", 1)[1].strip()
            elif line.lower().startswith("confiança:"):
                try:
                    conf = int("".join(ch for ch in line.split(":", 1)[1] if ch.isdigit()))
                except Exception:
                    conf = 0

        # 5) Sanitiza subgênero (se vier fora do universo conhecido, força "Não Identificado")
        if sub != "Subgênero Não Identificado" and sub not in SOFT_RULES:
            sub = "Subgênero Não Identificado"
            conf = 0

        # 6) Fallback heurístico se o GPT não identificar
        if sub == "Subgênero Não Identificado":
            fb_sub, fb_conf = backend_fallback_best_candidate(feats, cands)
            if fb_sub != "Subgênero Não Identificado":
                sub, conf = fb_sub, max(conf, fb_conf)

        return {
            "bpm": bpm_val,
            "subgenero": sub,
            "confidence": conf
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bpm": None,
            "subgenero": "Subgênero Não Identificado",
            "confidence": 0,
            "error": f"processing failed: {e.__class__.__name__}: {e}"
        })
