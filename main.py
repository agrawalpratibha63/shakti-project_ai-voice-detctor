# ==============================================================================
# AI VOICE DETECTOR - UNIVERSAL VERSION (FILE + BASE64 SUPPORT)
# ==============================================================================
import os
import base64
from fastapi import FastAPI, Request, HTTPException, Security, Depends
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

app = FastAPI(title="AI Voice Detector Universal", version="Universal.1.0")

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

async def get_api_key(
    request: Request,
    api_key_header: str = Security(api_key_header)
):
    # 1. Try Header
    if api_key_header == MATCH_API_KEY:
        return api_key_header
    
    # 2. Try Query Parameter (Example: ?api_key=shakti123) - Helpful for some tools
    query_key = request.query_params.get("api_key")
    if query_key == MATCH_API_KEY:
        return query_key

    raise HTTPException(
        status_code=403, 
        detail="❌ Access Denied: Invalid or Missing API Key"
    )

# ==============================================================================
# FEATURE EXTRACTION (UNTOUCHED)
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
# UNIVERSAL ENDPOINT (HANDLES FILE & BASE64)
# ==============================================================================
@app.post("/predict")
async def predict_endpoint(request: Request, api_key: str = Depends(get_api_key)):
    content_type = request.headers.get("content-type", "")
    y = None
    sr = None

    try:
        # CASE 1: JSON Input (Base64) - For Testing Tools
        if "application/json" in content_type:
            try:
                data = await request.json()
                b64_string = None
                
                # Try to find base64 string in any key
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 100:
                        b64_string = value
                        break
                
                if not b64_string:
                    raise HTTPException(400, "No Base64 audio found in JSON")
                
                # Fix padding if needed
                b64_string = b64_string.split(",")[-1] # Remove data:audio/mp3;base64, prefix if present
                missing_padding = len(b64_string) % 4
                if missing_padding:
                    b64_string += '=' * (4 - missing_padding)
                    
                audio_bytes = base64.b64decode(b64_string)
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
            except Exception as e:
                print(f"Base64 Error: {e}")
                raise HTTPException(400, "Invalid Base64 Audio Data")

        # CASE 2: File Upload - For Website
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(400, "No file uploaded")
            
            content = await file.read()
            y, sr = librosa.load(io.BytesIO(content), sr=SAMPLE_RATE)

        else:
            raise HTTPException(400, "Unsupported Content-Type. Use JSON (Base64) or Multipart (File).")

        # Process Audio
        if librosa.get_duration(y=y, sr=sr) < 0.5:
            raise HTTPException(400, "Audio too short.")
            
        result = process_audio(y, sr)
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Error: {str(e)}"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)