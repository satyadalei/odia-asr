print("Importing packages...")

print("Importing sounddevice...")
import sounddevice as sd
print("Imported sounddevice successfully.")

print("Importing numpy...")
import numpy as np
print("Imported numpy successfully.")

print("Importing torch...")
import torch
print("Imported torch successfully.")

print("Importing transformers...")
from transformers import AutoModelForCTC, AutoProcessor, AutoModel
print("Imported transformers successfully.")

print("Importing time...")
import time
print("Imported time successfully.")

print("Imported packages successfully.")
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANGUAGE_CODE = "or"

SAMPLE_RATE = 16000
DURATION = 3 # seconds per chunk

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
model.eval()

def record_chunk():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()
    return torch.tensor(audio).T #shape: [1, T]

print("🎙️ Listening... Press Ctrl+C to stop\n")

try:
    while True:
        audio = record_chunk()
        
        with torch.no_grad():
            text = model(audio, LANGUAGE_CODE, "ctc")

        if text.strip():
            print(f" 📝 {text}")
except KeyboardInterrupt:
    print("\n Recording stopped by user.")