#!/usr/bin/env python3
import whisper
import whisperx
from pyannote.audio import Pipeline
import torchaudio
import torch
import sys

# ---- Configuration ----
WHISPER_MODELS = ["medium", "large-v3"]  # list of Whisper models to preload
ALIGNMENT_LANG = "en"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN = sys.argv[1]

# ---- Device selection ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Using device: {device}")

# ---- 1. Preload Whisper models ----
for model_name in WHISPER_MODELS:
    print(f"[*] Preloading Whisper model '{model_name}'...")
    whisper.load_model(model_name, device=device)
    print(f"[✓] Whisper model '{model_name}' cached.")

# ---- 2. Preload WhisperX alignment model ----
print(f"[*] Preloading WhisperX alignment model for language '{ALIGNMENT_LANG}'...")
align_model, metadata = whisperx.load_align_model(language_code=ALIGNMENT_LANG, device=device)
print("[✓] WhisperX alignment model cached.")

# ---- 3. Preload Wav2Vec2 ASR model ----
print("[*] Preloading Wav2Vec2 ASR model...")
torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
print("[✓] Wav2Vec2 ASR model cached.")

# ---- 4. Preload Pyannote speaker diarization ----
print(f"[*] Preloading Pyannote diarization model '{DIARIZATION_MODEL}'...")
Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
print("[✓] Pyannote diarization model cached.")

print("[✓] All selected models preloaded and cached. Transcription can now run without network downloads.")
