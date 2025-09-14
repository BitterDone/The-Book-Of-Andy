#!/usr/bin/env python3
# Fully-Python script with speaker diarization
import argparse
import hashlib
import os
import subprocess
import feedparser
import requests
import whisperx
from pyannote.audio import Pipeline
from datetime import timedelta
import warnings
import math
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- CONFIG ----
TRANSCRIPTS_DIR = "original_transcripts"
# Noticed some accuracy issues with diarization on base. Moving to medium. Large-v3 is even better with a strong GPU.
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

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

def transcribe_with_speakers(model, audio_file: str, hf_token: str, fill_gaps: bool, device: str, detailed_logging: bool) -> str:
    """Run Whisper + diarization, keeping Whisper as ground truth timeline,
    and filling diarization gaps with Whisper fallback.
    """
    # Removed whisper in favor of whisperx
    # # Whisper with timestamps
    # result = model.transcribe(audio_file, word_timestamps=True)
    # Step 1: Transcribe with WhisperX
    # Documentation:
    # https://github.com/m-bain/whisperX/blob/2d9ce44329ae73af2520196d31cd14b6192ace44/whisperx/asr.py#L189
    result = model.transcribe(
        audio_file,
        language="en",
        # condition_on_previous_text=True,
        # suppress_blank=False  # <-- keep even low-energy start
    )

    if detailed_logging:
        print(f"[*] Transcribed, loading alignment model...")
    # Step 2: Load alignment model (language-specific)
    align_model, metadata = whisperx.load_align_model(
        # Results in
        # No language specified, language will be first be detected for each audio file (increases inference time).
        # language_code=result["language"], device=device
        language_code="en", device=device
    )

# Start alignment section ---------------------------------
# Notes:
#     Use "cpu" for device inside the worker processes. 
#       GPU isn’t shared easily across multiple Python processes.
#     Number of processes defaults to your CPU core count (os.cpu_count()), 
#       but you can set ProcessPoolExecutor(max_workers=8) for a limit.
#     Keep track of order if necessary — as_completed() returns results as they finish, 
#       so you may need to sort aligned_segments by start time afterward.

    aligned_segments = []   # not currently used since keeping TQDM progress bar
    word_segments = []      # used for eliminating timestamp gaps and better alignment
    # # Use this with TQDM progress bar and parallel processing
    def align_segment(seg, align_model, metadata, audio_file, device):
        result = whisperx.align([seg], align_model, metadata, audio_file, device)
        return result["segments"][0], result["word_segments"]
    
    max_workers = min(os.cpu_count(), 8)  # limit to 8 or your CPU count
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(align_segment, seg, align_model, metadata, audio_file, "cpu")
                for seg in result["segments"]]

        for future in as_completed(futures):
            seg_aligned, words = future.result()
            aligned_segments.append(seg_aligned) # not currently used
            word_segments.extend(words)

    # # Use this with TQDM progress bar and synchronous processing
    # for seg in tqdm(result["segments"], desc="Aligning segments"):
    #     aligned = whisperx.align([seg], align_model, metadata, audio_file, device)
    #     word_segments.extend(aligned["word_segments"])

    # # Reconstruct result_aligned like WhisperX would return
    result_aligned = {
        "segments": aligned_segments,   # not currently used since keeping TQDM progress bar
        "word_segments": word_segments  # used for eliminating timestamp gaps and better alignment
    }

    # # Use this without TQDM progress bar
    # # Step 3: Perform alignment for accurate word-level timestamps
    # result_aligned = whisperx.align(
    #     result["segments"], align_model, metadata, audio_file, device
    # )

# End alignment section ---------------------------------

    if detailed_logging:
        print(f"[*] Aligned, performing diarization...")
    # PyAnnote diarization
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    diarization = pipeline(audio_file)

    lines = []
    last_speaker = "UNKNOWN"

    # ---- OPTION 1: Whisper text always kept ----
    # for seg in result["segments"]: # original Whisper segments
    for seg in result_aligned["segments"]: # new whisperx segments
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()


        # Find diarization speaker that overlaps this whisper segment
        speaker = None
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= end and turn.end >= start:  # overlap condition
                speaker = spk
                break

        if speaker is None:
            # No diarization label → fallback
            speaker = last_speaker
            print(
                f"[!] Gap detected: {format_time(start)}–{format_time(end)} "
                f"→ assigning speaker={speaker}"
            )
        else:
            last_speaker = speaker

        lines.append(f"[{format_time(start)} - {format_time(end)}] {speaker}: {text}")

    if detailed_logging:
        print(f"[*] Diarized, filling gaps...")
    # ---- Option 3 extra: fill diarization gaps that Whisper didn’t cover ----
    # Walk through diarization timeline and insert dummy lines if Whisper missed it.
    if fill_gaps:
        whisper_start = result["segments"][0]["start"]
        whisper_end = result["segments"][-1]["end"]

        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.end < whisper_start or turn.start > whisper_end:
                continue  # outside whisper scope
            overlap = any(
                seg["start"] <= turn.end and seg["end"] >= turn.start
                for seg in result["segments"]
            )
            if not overlap:
                gap_line = (
                    f"[{format_time(turn.start)} - {format_time(turn.end)}] "
                    f"{spk}: [no Whisper transcript — diarization only]"
                )
                print(f"[!] Filling diarization-only gap: {gap_line}")
                lines.append(gap_line)

    # Keep transcript sorted by time
    def line_start_time(line: str) -> float:
        # extract HH:MM:SS from "[HH:MM:SS - ...]"
        timestamp = line.split("]")[0][1:].split(" - ")[0]
        h, m, s = map(int, timestamp.split(":"))
        return h * 3600 + m * 60 + s

    if detailed_logging:
        print(f"[*] Gapped, sorting by timestamp...")
    lines.sort(key=line_start_time)

    if detailed_logging:
        print(f"[*] Sorted, returning from transcribe_with_speakers()")
    return apply_corrections("\n".join(lines))

def transcribe(model, audio_file: str) -> str:
    """Run Whisper model and return transcript text"""
    result = model.transcribe(audio_file)
    return apply_corrections(result["text"])

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
    warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
    
    # Suppress low-level C++ warnings from PyTorch (like NNPACK)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rss", required=True, help="Podcast RSS feed URL")
    parser.add_argument("--repo", required=True, help="Path to local repo")
    parser.add_argument("--token", required=True, help="Hugging Face token for diarization model")
    parser.add_argument("--diarize", required=True, help="Control speaker diarization (on/off)")
    parser.add_argument("--fill-gaps", default="off", help="Fill diarization gaps with placeholders (on/off)")
    parser.add_argument("--detailed-logs", default="off", help="Fill diarization gaps with placeholders (on/off)")
    args = parser.parse_args()

    # Prepare output directory
    outdir = os.path.join(args.repo, TRANSCRIPTS_DIR)
    os.makedirs(outdir, exist_ok=True)

    # Load Whisper model once
    print(f"[*] Loading Whisper model: {WHISPER_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(WHISPER_MODEL, device)
    # # large-v3 requires ~10 GB VRAM minimum. An A100 (40GB) or H100 is safe.
    # # medium requires ~5 GB VRAM. Runs fine on cheaper GPUs like T4.
    # # If out-of-memory errors arise,
    # # fall back to "base" at WHISPER_MODEL = "medium"
    # # Or enable compute_type="int8" for quantization:
    # model = whisperx.load_model(WHISPER_MODEL, device, compute_type="int8")


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

        if args.diarize.lower() == "on":
            print(f"[*] Transcribing with speakers...")
            # Full transcription with diarization
            transcript = transcribe_with_speakers(model, clean_wav, args.token, fill_gaps=(args.fill_gaps.lower() == "on"), device=device, detailed_logging=(args.detailed_logs.lower() == "on"))
        else:
            print(f"[*] Transcribing without speakers...")
            # Simple transcription without diarization
            transcript = transcribe(model, clean_wav)

        print(f"[*] Writing file...")
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
