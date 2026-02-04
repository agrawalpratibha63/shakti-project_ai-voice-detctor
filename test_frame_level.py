import numpy as np
import librosa
from main import classify_audio

# Create a simple test signal
sr = 22050
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))
test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

print('Testing frame-level classification with normalization...')
result = classify_audio(test_signal, sr)
print('Prediction:', result['prediction'])
print('Confidence:', result['confidence'])
print('Frames processed:', result['detailed_scores']['total_frames'])
print('AI votes:', result['detailed_scores']['ai_votes'])
print('Human votes:', result['detailed_scores']['human_votes'])
print('Test completed successfully!')