#!/usr/bin/env python3
# Fully-Python script with speaker diarization
import argparse
import hashlib
import os
import subprocess
import feedparser
import requests
import whisper
from pyannote.audio import Pipeline
from datetime import timedelta
import warnings

# ---- CONFIG ----
TRANSCRIPTS_DIR = "original_transcripts"
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

def hash_guid(guid: str) -> str:
    """Stable short ID from RSS GUID or URL"""
    return hashlib.sha1(guid.encode()).hexdigest()[:12]

def download_audio(url: str, dest: str):
    """Download audio file from RSS enclosure URL"""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def clean_audio(infile: str, outfile: str):
    """Convert audio to mono 16kHz WAV for Whisper"""
    subprocess.run([
        "ffmpeg", "-y", "-i", infile,
        "-ac", "1", "-ar", "16000", outfile
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def transcribe_with_speakers(model, audio_file: str, hf_token: str) -> str:
    """Run Whisper + diarization and return speaker-labeled transcript"""

    # Whisper with timestamps
    result = model.transcribe(audio_file, word_timestamps=True)

    # PyAnnote diarization
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    diarization = pipeline(audio_file)

    lines = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Collect Whisper text segments that fall into this diarization window
        segment_texts = [
            seg["text"]
            for seg in result["segments"]
            if seg["start"] >= turn.start and seg["end"] <= turn.end
        ]
        text = " ".join(segment_texts).strip()
        if text:
            lines.append(
                f"[{format_time(turn.start)} - {format_time(turn.end)}] {speaker}: {text}"
            )

    return "\n".join(lines)

def transcribe(model, audio_file: str) -> str:
    """Run Whisper model and return transcript text"""
    result = model.transcribe(audio_file)
    return result["text"]

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
    warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rss", required=True, help="Podcast RSS feed URL")
    parser.add_argument("--repo", required=True, help="Path to local repo")
    parser.add_argument("--token", required=True, help="Hugging Face token for diarization model")
    parser.add_argument("--diarize", required=True, help="Control speaker diarization (on/off)")
    args = parser.parse_args()

    # Prepare output directory
    outdir = os.path.join(args.repo, TRANSCRIPTS_DIR)
    os.makedirs(outdir, exist_ok=True)

    # Load Whisper model once
    print(f"[*] Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)

    # Parse RSS feed
    feed = feedparser.parse(args.rss)

    for entry in feed.entries:
        guid = entry.get("id") or entry.link
        title = entry.title.replace("/", "-").replace(" ", "_")
        fname_base = f"{hash_guid(guid)}_{title[:50]}"
        txt_path = os.path.join(outdir, fname_base + ".txt")

        if os.path.exists(txt_path):
            print(f"[*] Skipping {title} (already transcribed).")
            continue

        audio_url = entry.enclosures[0].href
        raw_audio = os.path.join(outdir, fname_base + ".mp3")
        clean_wav = os.path.join(outdir, fname_base + ".wav")

        print(f"[*] Downloading: {title}")
        download_audio(audio_url, raw_audio)

        print(f"[*] Cleaning audio...")
        clean_audio(raw_audio, clean_wav)

        print(f"[*] Transcribing with speakers...")
        if args.diarize.lower() == "on":
            # Full transcription with diarization
            transcript = transcribe_with_speakers(model, clean_wav, args.token)
        else:
            # Simple transcription without diarization
            transcript = transcribe(model, clean_wav)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {entry.title}\n")
            f.write(f"Date: {entry.get('published', 'unknown')}\n")
            f.write(f"GUID: {guid}\n\n")
            f.write(transcript.strip() + "\n")

        # Cleanup
        os.remove(raw_audio)
        os.remove(clean_wav)

        print(f"[✓] Saved transcript: {txt_path}")

if __name__ == "__main__":
    main()
