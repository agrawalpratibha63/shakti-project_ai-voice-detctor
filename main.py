# ==============================================================================
# AI VOICE DETECTOR - JSON OUTPUT VERSION
# ==============================================================================
from fastapi import FastAPI, File, UploadFile, HTTPException
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

app = FastAPI(title="AI Voice Detector", version="JSON.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 22050

# ==============================================================================
# FEATURE EXTRACTION (PERFECT ORIGINAL)
# ==============================================================================
def calculate_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    try:
        y, _ = librosa.effects.trim(audio, top_db=20)
        if len(y) < 512: return {}
        
        # 1. Pitch Analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        f0_std = np.std(f0) if len(f0) > 0 else 0.0

        # 2. Spectral Analysis
        S = np.abs(librosa.stft(y))
        S_norm = S / (np.sum(S, axis=0) + 1e-9)
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-9), axis=0)
        mean_entropy = np.mean(entropy)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = np.std(zcr)

        return {
            "f0_std": float(f0_std),
            "entropy": float(mean_entropy),
            "zcr_std": float(zcr_std)
        }
    except:
        return {}

# ==============================================================================
# AI DETECTION LOGIC (UNTOUCHED)
# ==============================================================================
def analyze_frame(features: Dict[str, float]) -> str:
    score = 0
    
    if features['f0_std'] > 5.0: score += 2
    elif features['f0_std'] < 1.5: score -= 3
    
    if features['entropy'] > 3.0: score += 1
    elif features['entropy'] < 1.5: score -= 2
    
    if features['zcr_std'] > 0.05: score += 1
    elif features['zcr_std'] < 0.01: score -= 2

    return "Human" if score >= 0 else "AI"

# ==============================================================================
# MAIN PROCESSOR
# ==============================================================================
def process_audio(y: np.ndarray, sr: int):
    chunk_size = int(0.5 * sr)
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    
    ai_votes = 0
    human_votes = 0
    total_valid_frames = 0
    
    for chunk in chunks:
        if len(chunk) < chunk_size: continue
        feats = calculate_features(chunk, sr)
        if not feats: continue
        
        prediction = analyze_frame(feats)
        if prediction == "AI": ai_votes += 1
        else: human_votes += 1
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

    # Returning Raw JSON Data
    return {
        "prediction": final_pred,
        "confidence": round(confidence, 2),
        "reason": f"Analyzed {total_valid_frames} frames. AI Score: {int(ai_percent*100)}%"
    }

# ==============================================================================
# API ENDPOINT
# ==============================================================================
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
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
    print("Server running on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)