#!/usr/bin/env python3

import whisperx
from pyannote.audio import Pipeline
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# # Use this with TQDM progress bar and parallel processing
# Move here to avoid
# AttributeError: Can't pickle local object 'transcribe_with_speakers.<locals>.align_segment'
def align_segment(seg, align_model, metadata, audio_file, device):
    result = whisperx.align([seg], align_model, metadata, audio_file, device)
    return result["segments"][0], result["word_segments"]
    
def transcribe_with_speakers(model, audio_file: str, hf_token: str, fill_gaps: bool, detailed_logs: bool) -> str:
    """Run Whisper + diarization, keeping Whisper as ground truth timeline,
    and filling diarization gaps with Whisper fallback.
    """
    
    print(f"[✓] detailed_logs: {detailed_logs}")
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

    if detailed_logs:
        print(f"[*] Transcribed, loading alignment model...")
    # Step 2: Load alignment model (language-specific)
    align_model, metadata = whisperx.load_align_model(
        # Results in
        # No language specified, language will be first be detected for each audio file (increases inference time).
        # language_code=result["language"], device=device
        language_code=ALIGN_LANG, device=device
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
    # align_segment is defined above to avoid 
    # AttributeError: Can't pickle local object 'transcribe_with_speakers.<locals>.align_segment'
   
    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures = [
            executor.submit(align_segment, seg, align_model, metadata, audio_file, "cpu")
                for seg in result["segments"]
            ]

        for future in as_completed(futures):
            seg_aligned, words = future.result()
            aligned_segments.append(seg_aligned) # not currently used
            word_segments.extend(words)

    # # Use this with TQDM progress bar and synchronous processing
    # for seg in tqdm(result["segments"], desc="Aligning segments"):
    #     aligned = whisperx.align([seg], align_model, metadata, audio_file, device)
    #     word_segments.extend(aligned["word_segments"])

# Below is duplicated into transcribe_with_speakers_parellel_align
    # # Reconstruct result_aligned like WhisperX would return
    result_aligned = {
        "segments": aligned_segments,   # not currently used since keeping TQDM progress bar
        "word_segments": word_segments  # used for eliminating timestamp gaps and better alignment
    }
    
    print(len(aligned_segments)) 
    print(len(word_segments))

    # # Use this without TQDM progress bar
    # # Step 3: Perform alignment for accurate word-level timestamps
    # result_aligned = whisperx.align(
    #     result["segments"], align_model, metadata, audio_file, device
    # )

# End alignment section ---------------------------------

    if detailed_logs:
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

    if detailed_logs:
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

def transcribe(model, audio_file: str, apply_corrections) -> str:
    """Run Whisper model and return transcript text"""
    result = model.transcribe(audio_file)
    return apply_corrections(result["text"])
