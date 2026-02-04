# ==============================================================================
# AI VOICE DETECTOR - SECURE API KEY VERSION
# ==============================================================================
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import librosa
import io
import warnings
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="AI Voice Detector Secure", version="Secure.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 22050

# --- SECURITY CONFIGURATION ---
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Render से API Key उठाएगा, अगर वहां नहीं मिली तो डिफॉल्ट "shakti123" होगी
MATCH_API_KEY = os.getenv("API_KEY", "shakti123")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == MATCH_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, 
            detail="❌ Access Denied: Invalid or Missing API Key"
        )

# ==============================================================================
# FEATURE EXTRACTION (SAME PERFECT LOGIC)
# ==============================================================================
def calculate_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    try:
        y, _ = librosa.effects.trim(audio, top_db=20)
        if len(y) < 512: return {}
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        f0_std = np.std(f0) if len(f0) > 0 else 0.0

        S = np.abs(librosa.stft(y))
        S_norm = S / (np.sum(S, axis=0) + 1e-9)
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-9), axis=0)
        mean_entropy = np.mean(entropy)
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = np.std(zcr)

        return {"f0_std": float(f0_std), "entropy": float(mean_entropy), "zcr_std": float(zcr_std)}
    except:
        return {}

def analyze_frame(features: Dict[str, float]) -> str:
    score = 0
    if features['f0_std'] > 5.0: score += 2
    elif features['f0_std'] < 1.5: score -= 3
    
    if features['entropy'] > 3.0: score += 1
    elif features['entropy'] < 1.5: score -= 2
    
    if features['zcr_std'] > 0.05: score += 1
    elif features['zcr_std'] < 0.01: score -= 2

    return "Human" if score >= 0 else "AI"

def process_audio(y: np.ndarray, sr: int):
    chunk_size = int(0.5 * sr)
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    ai_votes = 0
    total_valid_frames = 0
    
    for chunk in chunks:
        if len(chunk) < chunk_size: continue
        feats = calculate_features(chunk, sr)
        if not feats: continue
        if analyze_frame(feats) == "AI": ai_votes += 1
        total_valid_frames += 1
    
    if total_valid_frames == 0:
        return {"prediction": "Unknown", "confidence": 0.0, "reason": "Audio too short."}
    
    ai_percent = ai_votes / total_valid_frames
    if ai_percent > 0.60:
        final_pred = "STRONG AI-GENERATED VOICE" if ai_percent > 0.8 else "PROBABLY AI-GENERATED VOICE"
        confidence = ai_percent
    else:
        final_pred = "STRONG HUMAN VOICE" if (1-ai_percent) > 0.8 else "POSSIBLY HUMAN VOICE"
        confidence = 1 - ai_percent

    return {
        "prediction": final_pred,
        "confidence": round(confidence, 2),
        "reason": f"Analyzed {total_valid_frames} frames. AI Score: {int(ai_percent*100)}%"
    }

# ==============================================================================
# SECURE API ENDPOINT
# ==============================================================================
@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key)  # <--- LOCK IS HERE
):
    if not file.filename.lower().endswith(('.mp3', '.mpeg')):
        raise HTTPException(400, "Invalid format. Use MP3 or MPEG only.")
    
    try:
        content = await file.read()
        y, sr = librosa.load(io.BytesIO(content), sr=SAMPLE_RATE)
        if librosa.get_duration(y=y, sr=sr) < 0.5:
            raise HTTPException(400, "Audio too short.")
        result = process_audio(y, sr)
        return result
    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error."})

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)