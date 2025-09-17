#!/usr/bin/env python3
# Fully-Python script with speaker diarization

# Suppress low-level C++ warnings from PyTorch (like NNPACK)
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import argparse
import feedparser
import whisperx
from pyannote.audio import Pipeline
import warnings
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.helpers import apply_corrections, hash_guid, download_audio, clean_audio, format_time, split_audio_to_chunks
from slimfile import transcribe, transcribe_with_speakers

# ---- CONFIG ----
TRANSCRIPTS_DIR = "original_transcripts"
# Noticed some accuracy issues with diarization on base. Moving to medium. Large-v3 is even better with a strong GPU.
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ALIGN_LANG="en"
MAX_WORKERS = min(os.cpu_count(), 8)  # limit to 8 or your CPU count

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_chunk(chunk_file, chunk_id):
    # Load models inside the process to avoid pickling issues
    # Load Whisper model once
    print(f"[*] Chunk #{chunk_id} Loading Whisper model: {WHISPER_MODEL}")
    model = whisperx.load_model(WHISPER_MODEL, device)
    # # large-v3 requires ~10 GB VRAM minimum. An A100 (40GB) or H100 is safe.
    # # medium requires ~5 GB VRAM. Runs fine on cheaper GPUs like T4.
    # # If out-of-memory errors arise,
    # # fall back to "base" at WHISPER_MODEL = "medium"
    # # Or enable compute_type="int8" for quantization:
    # model = whisperx.load_model(WHISPER_MODEL, device, compute_type="int8")

    align_model, metadata = whisperx.load_align_model(language_code=ALIGN_LANG, device=device)

    # Transcribe chunk
    print(f"[*] Transcribing chunk #{chunk_id}")
    result = model.transcribe(chunk_file, language="en")
    print(f"[*] Transcribed chunk #{chunk_id}")

    # Align segments
    word_segments = []
    aligned_segments = []
    for seg in result["segments"]:
        aligned = whisperx.align([seg], align_model, metadata, chunk_file, device)
        aligned_segments.append(aligned["segments"][0])
        word_segments.extend(aligned["word_segments"])

    return aligned_segments, word_segments
    
def transcribe_with_speakers_parellel_align(model, audio_file: str, hf_token: str, fill_gaps: bool, detailed_logs: bool) -> str:
    chunks = split_audio_to_chunks(audio_file)

    aligned_segments_all = []
    word_segments_all = []

    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk, idx) for idx, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            aligned_segments, word_segments = future.result()
            aligned_segments_all.extend(aligned_segments)
            word_segments_all.extend(word_segments)

    # Sort by start time
    aligned_segments_all.sort(key=lambda s: s["start"])
    word_segments_all.sort(key=lambda w: w["start"])

    # Cleanup chunk files
    for chunk in chunks:
        os.remove(chunk)

    result_aligned = {
        "segments": aligned_segments_all,
        "word_segments": word_segments_all
    }
    
    print(len(aligned_segments_all)) # 282
    print(len(word_segments_all)) # 18364

    if detailed_logs:
        print(f"[*] Aligned, performing diarization...")
    # PyAnnote diarization
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    diarization = pipeline(audio_file)

    lines = []
    last_speaker = "UNKNOWN"

    for seg in tqdm(result_aligned["segments"], desc="Diarizing segments"):
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
            # tqdm.write(
            #     f"[!] Gap detected: {format_time(start)}–{format_time(end)} "
            #     f"→ assigning speaker={speaker}"
            # )
        else:
            last_speaker = speaker

        lines.append(f"[{format_time(start)} - {format_time(end)}] {speaker}: {text}")

    if detailed_logs:
        print(f"[*] Diarized, filling gaps...")
    # Walk through diarization timeline and insert dummy lines if Whisper missed it.
    if fill_gaps:
        whisper_start = result_aligned["segments"][0]["start"]
        whisper_end = result_aligned["segments"][-1]["end"]

        turns = list(diarization.itertracks(yield_label=True))
        for turn, _, spk in tqdm(turns, desc="Filling diarization gaps"):
            if turn.end < whisper_start or turn.start > whisper_end:
                continue  # outside whisper scope
            overlap = any(
                seg["start"] <= turn.end and seg["end"] >= turn.start
                for seg in result_aligned["segments"]
            )
            if not overlap:
                gap_line = (
                    f"[{format_time(turn.start)} - {format_time(turn.end)}] "
                    f"{spk}: [no Whisper transcript — diarization only]"
                )
                # tqdm.write(f"[!] Filling diarization-only gap: {gap_line}")
                lines.append(gap_line)

    print(len(lines)) 
    # Keep transcript sorted by time
    def line_start_time(line: str) -> float:
        # extract HH:MM:SS from "[HH:MM:SS - ...]"
        timestamp = line.split("]")[0][1:].split(" - ")[0]
        h, m, s = map(int, timestamp.split(":"))
        return h * 3600 + m * 60 + s

    if detailed_logs:
        print(f"[*] Gapped, sorting by timestamp...")
    lines.sort(key=line_start_time)

    if detailed_logs:
        print(f"[*] Sorted, returning from transcribe_with_speakers()")
    print(len(lines)) 
    return apply_corrections("\n".join(lines))

def start_process(args, outdir, model):
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

        if os.path.exists(raw_audio) or os.path.exists(clean_wav):
            print(f"[*] Skipping download for {title} (audio already exists).")
        else:
            print(f"[*] Downloading: {title}")
            download_audio(audio_url, raw_audio)

        print(f"[*] Cleaning audio...")
        clean_audio(raw_audio, clean_wav)

        if args.diarize.lower() == "on":
            print(f"[*] Transcribing with speakers...")
            # transcript = transcribe_with_speakers(model, clean_wav, args.token, fill_gaps=(args.fill_gaps.lower() == "on"), detailed_logs=(args.detailed_logs.lower() == "on"))
            transcript = transcribe_with_speakers_parellel_align("", clean_wav, args.token, fill_gaps=(args.fill_gaps.lower() == "on"), detailed_logs=(args.detailed_logs.lower() == "on"))
        else:
            print(f"[*] Transcribing without speakers...")
            transcript = transcribe(model, clean_wav, apply_corrections = apply_corrections)

        print(f"[*] Writing file...")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {entry.title}\n")
            f.write(f"Date: {entry.get('published', 'unknown')}\n")
            f.write(f"GUID: {guid}\n\n")
            f.write(transcript.strip() + "\n")

        # os.remove(raw_audio)
        # os.remove(clean_wav)

        print(f"[✓] Saved transcript: {txt_path}")

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
    parser.add_argument("--fill-gaps", default="off", help="Fill diarization gaps with placeholders (on/off)")
    parser.add_argument("--detailed-logs", default="off", help="Fill diarization gaps with placeholders (on/off)")
    args = parser.parse_args()

    outdir = os.path.join(args.repo, TRANSCRIPTS_DIR)
    os.makedirs(outdir, exist_ok=True)

    # # Can't load the model out here if using ProcessPoolExecutor
    # print(f"[*] Loading Whisper model: {WHISPER_MODEL}")
    # model = whisperx.load_model(WHISPER_MODEL, device)
    # # # large-v3 requires ~10 GB VRAM minimum. An A100 (40GB) or H100 is safe.
    # # # medium requires ~5 GB VRAM. Runs fine on cheaper GPUs like T4.
    # # # If out-of-memory errors arise, fall back to "base" at WHISPER_MODEL = "medium"
    # # # Or enable compute_type="int8" for quantization:
    # # model = whisperx.load_model(WHISPER_MODEL, device, compute_type="int8")

    start_process(args, outdir, model=None)

if __name__ == "__main__":
    main()
