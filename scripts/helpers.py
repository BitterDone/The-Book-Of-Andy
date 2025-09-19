#!/usr/bin/env python3

import hashlib
import subprocess
import requests
from datetime import timedelta
import math
from pydub import AudioSegment
import threading
import time
from tqdm import tqdm
import torchaudio

AUDIO_FILE_CHUNK_LENGTH_MS = 60 * 1000 * 10  # 10 min

# Common misheard phrase corrections
COMMON_FIXES = {
    "smoking mirrors": "smoke and mirrors",
    "jury box": "jewelry box",
    "booted slow": "booty swole",
    "Priscilla": "Frisella",
    # add more as you encounter them
}

def apply_corrections(text: str) -> str:
    """Apply common misheard phrase corrections to transcript text"""
    for wrong, right in COMMON_FIXES.items():
        # case-insensitive replace
        text = text.replace(wrong, right)
        text = text.replace(wrong.capitalize(), right.capitalize())
    return text

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
    """Convert audio to mono 16kHz WAV for Whisper, pad start/end to keep intro/outro"""
    subprocess.run([
        "ffmpeg", "-y", "-i", infile,
        "-ac", "1", "-ar", "16000",
        "-af", "apad=pad_dur=2",  # pad 2 seconds of silence at start and end
                                  # increase pad_dur if intros are longer or quieter.
        outfile
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def split_audio_to_chunks(audio_path, chunk_length_ms=AUDIO_FILE_CHUNK_LENGTH_MS):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk_file = f"{audio_path}_chunk{i//1000}.wav"
        audio[i:i+chunk_length_ms].export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def run_with_progress(pipeline, audio_file, desc="Running diarization", speed_factor=0.25):
    """
    Wraps PyAnnote pipeline(audio_file) with a tqdm progress bar.

    Args:
        pipeline: The PyAnnote pipeline object
        audio_file: Path to audio file
        desc: Label for the tqdm bar
        speed_factor: Rough multiplier (smaller = faster relative processing).
                    Adjust based on your system (try 0.2–0.5).

    Returns:
        diarization result
    """
    # Estimate duration (in seconds) of the audio file
    info = torchaudio.info(audio_file)
    duration = info.num_frames / info.sample_rate

    # Estimate how long diarization will take
    est_time = duration * speed_factor

    result = {}

    def run_pipeline():
        result["diarization"] = pipeline(audio_file)

    # Launch pipeline in background
    t = threading.Thread(target=run_pipeline)
    t.start()

    # Progress bar (ticks in 1-second steps)
    with tqdm(total=int(est_time), desc=desc, unit="s", dynamic_ncols=True) as pbar:
        elapsed = 0
        while t.is_alive():
            time.sleep(1)
            elapsed += 1
            pbar.update(1 if elapsed <= est_time else 0)
        t.join()
        pbar.n = pbar.total  # force complete
        pbar.close()

    return result["diarization"]