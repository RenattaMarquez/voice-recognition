import os
import pickle
from audio_recorder import AudioRecorder
from recognition_utils import recognize  

# Load models
with open("models/quantizer.pkl", "rb") as f:
    quantizer = pickle.load(f)
with open("models/hmm_models.pkl", "rb") as f:
    hmm_models = pickle.load(f)

# Initialize recorder
recorder = AudioRecorder(folder="command", duration=2)

# Record audio and save to file
recorded_path = recorder.record("live_test.wav")

# Run recognition
recognize(recorded_path, quantizer, hmm_models)
