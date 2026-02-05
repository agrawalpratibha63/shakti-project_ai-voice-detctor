# ==============================================================================
# AI VOICE DETECTOR - UNIVERSAL FIX (ACCEPTS ANY KEY)
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

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == MATCH_API_KEY:
        return api_key_header
    else:
        # अगर टेस्टर बिना Key के ट्राई करे, तो भी शायद पास कर दें (Debugging के लिए)
        # लेकिन अभी सुरक्षा रखते हैं।
        raise HTTPException(
            status_code=403, 
            detail="❌ Access Denied: Invalid or Missing API Key"
        )

# ==============================================================================
# FEATURE EXTRACTION & LOGIC
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
# UNIVERSAL PREDICT ENDPOINT (Fixes 422 Error)
# ==============================================================================
@app.post("/predict")
async def predict_endpoint(
    request: Request, 
    api_key: str = Depends(get_api_key)
):
    try:
        # 1. डेटा प्राप्त करें (चाहे JSON हो या कुछ और)
        try:
            body_data = await request.json()
        except:
            # अगर JSON नहीं है, तो शायद सीधा टेक्स्ट भेजा है
            raw = await request.body()
            body_data = {"raw": raw.decode('utf-8', errors='ignore')}

        # 2. Base64 स्ट्रिंग ढूंढें (चाहे किसी भी Key में हो)
        base64_string = None
        
        if isinstance(body_data, dict):
            # हम Dictionary की सारी Values चेक करेंगे
            for key, value in body_data.items():
                # अगर कोई लम्बी स्ट्रिंग मिली, तो वही ऑडियो है
                if isinstance(value, str) and len(value) > 100:
                    base64_string = value
                    break
        
        if not base64_string:
             raise HTTPException(400, "Could not find any Base64 audio data in the request.")

        # 3. सफाई करें (Remove headers like data:audio/mp3;base64,...)
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # एक्स्ट्रा सफाई (New Lines हटाने के लिए)
        base64_string = base64_string.replace("\n", "").replace("\r", "").strip()

        # 4. डिकोड करें
        try:
            audio_bytes = base64.b64decode(base64_string)
        except Exception:
            raise HTTPException(400, "Invalid Base64 string found.")

        # 5. Librosa से लोड करें
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        
        # 6. प्रोसेस करें
        if librosa.get_duration(y=y, sr=sr) < 0.5:
            raise HTTPException(400, "Audio is too short (less than 0.5s).")
            
        result = process_audio(y, sr)
        return result
        
    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Processing Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)