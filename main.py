# ==============================================================================
# AI VOICE DETECTOR - BASE64 & FILE SUPPORT (HYBRID)
# ==============================================================================
import os
import base64
import io
import numpy as np
import librosa
import warnings
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="AI Voice Detector Base64", version="Base64.1.0")

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
MATCH_API_KEY = os.getenv("API_KEY", "shakti123")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == MATCH_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, 
            detail="❌ Access Denied: Invalid or Missing API Key"
        )

# --- REQUEST MODEL FOR BASE64 ---
class AudioRequest(BaseModel):
    audio_data: str  # This will hold the long Base64 string

# ==============================================================================
# FEATURE EXTRACTION & LOGIC (UNTOUCHED)
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
# NEW API ENDPOINT (SUPPORTS BASE64 JSON)
# ==============================================================================
@app.post("/predict")
async def predict_endpoint(
    request: AudioRequest,  # Now accepting JSON Body
    api_key: str = Depends(get_api_key)
):
    try:
        # 1. Get the Base64 string
        base64_string = request.audio_data
        
        # 2. Clean the string (Remove header like "data:audio/mp3;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        # 3. Decode Base64 to Bytes
        try:
            audio_bytes = base64.b64decode(base64_string)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 Audio Data")

        # 4. Load audio using Librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        
        # 5. Check Duration
        if librosa.get_duration(y=y, sr=sr) < 0.5:
            raise HTTPException(400, "Audio too short.")
            
        # 6. Process
        result = process_audio(y, sr)
        return result
        
    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Processing Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)