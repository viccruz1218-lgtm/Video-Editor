#!/usr/bin/env python3
"""
clean_video.py — Automatically removes filler words from interview footage.

Usage:
    python clean_video.py --input agent_interview.mp4 [--model base] [--padding 0.05]
"""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment
from pydub.effects import normalize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILLER_WORDS = {
    "um", "uh", "like", "so", "literally", "basically", "right", "okay",
    "kind", "sort", "mean",  # partial matches — see FILLER_PHRASES too
}

# Multi-word fillers — checked against consecutive word sequences
FILLER_PHRASES = [
    "you know",
    "i mean",
    "kind of",
    "sort of",
]

# Single-word fillers that are also real content words — only safe at sentence
# start or surrounded by longer pauses; anything else is flagged.
AMBIGUOUS_FILLERS = {"like", "so", "right", "okay", "basically", "literally"}

CROSSFADE_DURATION = 0.2   # seconds — audio crossfade on each cut
MAX_PAUSE_TO_KEEP = 0.3    # pauses ≤ this are left alone
MIN_PAUSE_TO_CUT = 0.5     # pauses > this may be trimmed
SENTENCE_BOUNDARY_PAUSE = 0.4  # gap that likely signals a sentence boundary


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float
    is_filler: bool = False
    flagged: bool = False
    flag_reason: str = ""


@dataclass
class Cut:
    start: float
    end: float
    reason: str
    flagged: bool = False
    flag_reason: str = ""


@dataclass
class Summary:
    filler_cuts: int = 0
    time_saved: float = 0.0
    flagged: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Transcription
# ---------------------------------------------------------------------------

def transcribe(video_path: str, model_name: str) -> list[Word]:
    """Run Whisper with word-level timestamps and return a flat word list."""
    print(f"[1/4] Transcribing with Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        verbose=False,
    )

    words: list[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            clean = re.sub(r"[^\w\s']", "", w["word"]).strip().lower()
            if clean:
                words.append(Word(text=clean, start=w["start"], end=w["end"]))

    print(f"    Transcribed {len(words)} words.")
    return words


# ---------------------------------------------------------------------------
# Step 2 — Filler detection
# ---------------------------------------------------------------------------

def is_sentence_start(words: list[Word], idx: int) -> bool:
    """True if this word is at the start of a sentence (large preceding pause)."""
    if idx == 0:
        return True
    gap = words[idx].start - words[idx - 1].end
    return gap >= SENTENCE_BOUNDARY_PAUSE


def is_sentence_end(words: list[Word], idx: int) -> bool:
    """True if the next word begins after a sentence-boundary pause."""
    if idx >= len(words) - 1:
        return True
    gap = words[idx + 1].start - words[idx].end
    return gap >= SENTENCE_BOUNDARY_PAUSE


def detect_fillers(words: list[Word]) -> list[Word]:
    """
    Mark each word as a filler or flag it for manual review.
    Multi-word phrases are handled first; single words second.
    """
    print("[2/4] Detecting filler words and phrases …")
    n = len(words)
    skip_until = -1

    for i, w in enumerate(words):
        if i <= skip_until:
            continue

        # --- multi-word phrase check ---
        for phrase in FILLER_PHRASES:
            parts = phrase.split()
            if i + len(parts) > n:
                continue
            if all(words[i + j].text == parts[j] for j in range(len(parts))):
                # Mark entire phrase
                at_start = is_sentence_start(words, i)
                after_end = is_sentence_end(words, i + len(parts) - 1)
                for j in range(len(parts)):
                    words[i + j].is_filler = True
                skip_until = i + len(parts) - 1
                # Ambiguous only if it's mid-sentence and not isolated
                if not at_start and not after_end and phrase in ("you know", "i mean"):
                    for j in range(len(parts)):
                        words[i + j].flagged = True
                        words[i + j].flag_reason = (
                            f"Phrase '{phrase}' mid-sentence — may affect meaning"
                        )
                break
        else:
            # --- single-word check ---
            if w.text in FILLER_WORDS:
                at_start = is_sentence_start(words, i)
                after_end = is_sentence_end(words, i)

                if w.text in AMBIGUOUS_FILLERS:
                    if at_start or after_end:
                        w.is_filler = True
                    else:
                        # Mid-sentence ambiguous word — flag for review
                        w.is_filler = True
                        w.flagged = True
                        w.flag_reason = (
                            f"'{w.text}' mid-sentence — verify removal doesn't break grammar"
                        )
                else:
                    # Unambiguous fillers (um, uh) — always remove
                    w.is_filler = True

    total = sum(1 for w in words if w.is_filler)
    flagged = sum(1 for w in words if w.flagged)
    print(f"    Found {total} filler instances ({flagged} flagged for review).")
    return words


# ---------------------------------------------------------------------------
# Step 3 — Build cut list
# ---------------------------------------------------------------------------

def merge_adjacent(cuts: list[Cut], gap_threshold: float = 0.05) -> list[Cut]:
    """Merge cuts that are very close together into a single cut."""
    if not cuts:
        return cuts
    merged = [cuts[0]]
    for c in cuts[1:]:
        last = merged[-1]
        if c.start - last.end <= gap_threshold:
            merged[-1] = Cut(
                start=last.start,
                end=max(last.end, c.end),
                reason=last.reason + " + " + c.reason,
                flagged=last.flagged or c.flagged,
                flag_reason=last.flag_reason or c.flag_reason,
            )
        else:
            merged.append(c)
    return merged


def build_cuts(words: list[Word], video_duration: float) -> list[Cut]:
    """
    Convert marked words into time-range cuts, including surrounding silence.
    Also trim long pauses (> MIN_PAUSE_TO_CUT) between kept segments.
    """
    print("[3/4] Building cut list …")
    cuts: list[Cut] = []

    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        if not w.is_filler:
            i += 1
            continue

        # Collect consecutive filler words
        j = i
        while j < n and words[j].is_filler:
            j += 1

        group = words[i:j]
        flagged = any(g.flagged for g in group)
        flag_reason = next((g.flag_reason for g in group if g.flag_reason), "")

        # Expand cut to include leading silence (up to the previous word end)
        cut_start = group[0].start
        if i > 0:
            prev_end = words[i - 1].end
            lead_silence = cut_start - prev_end
            if lead_silence > MAX_PAUSE_TO_KEEP:
                cut_start = prev_end + MAX_PAUSE_TO_KEEP

        # Expand cut to include trailing silence (up to next word start)
        cut_end = group[-1].end
        if j < n:
            next_start = words[j].start
            trail_silence = next_start - cut_end
            if trail_silence > MAX_PAUSE_TO_KEEP:
                cut_end = cut_end + MAX_PAUSE_TO_KEEP

        cuts.append(Cut(
            start=cut_start,
            end=cut_end,
            reason=f"filler: {' '.join(g.text for g in group)}",
            flagged=flagged,
            flag_reason=flag_reason,
        ))
        i = j

    # Also trim excessively long pauses between kept segments
    kept_words = [w for w in words if not w.is_filler]
    for idx in range(len(kept_words) - 1):
        gap_start = kept_words[idx].end
        gap_end = kept_words[idx + 1].start
        gap = gap_end - gap_start
        if gap > MIN_PAUSE_TO_CUT:
            # Leave MAX_PAUSE_TO_KEEP of silence, cut the rest
            trim_start = gap_start + MAX_PAUSE_TO_KEEP
            trim_end = gap_end
            if trim_end - trim_start > 0.05:
                cuts.append(Cut(
                    start=trim_start,
                    end=trim_end,
                    reason=f"long pause ({gap:.2f}s)",
                ))

    cuts.sort(key=lambda c: c.start)
    cuts = merge_adjacent(cuts)

    # Clamp to video bounds
    cuts = [c for c in cuts if c.start < video_duration]
    for c in cuts:
        c.end = min(c.end, video_duration)

    print(f"    Built {len(cuts)} cuts.")
    return cuts


# ---------------------------------------------------------------------------
# Step 4 — Export
# ---------------------------------------------------------------------------

def apply_audio_crossfade(
    segment_audio: AudioSegment,
    next_segment_audio: AudioSegment,
    fade_ms: int,
) -> tuple[AudioSegment, AudioSegment]:
    """Apply crossfade between two adjacent segments."""
    fade_ms = min(fade_ms, len(segment_audio) // 2, len(next_segment_audio) // 2)
    if fade_ms <= 0:
        return segment_audio, next_segment_audio
    seg_faded = segment_audio.fade_out(fade_ms)
    next_faded = next_segment_audio.fade_in(fade_ms)
    return seg_faded, next_faded


def export_video(
    input_path: str,
    cuts: list[Cut],
    output_path: str,
    crossfade_s: float = CROSSFADE_DURATION,
) -> Summary:
    """
    Render the output video by:
      1. Collecting kept time intervals (inverse of cuts)
      2. Applying audio crossfades between clips using pydub
      3. Concatenating video segments with moviepy
      4. Muxing the crossfaded audio back in
    """
    print("[4/4] Exporting cleaned video …")

    video = VideoFileClip(input_path)
    duration = video.duration

    # Build kept intervals
    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for c in cuts:
        if c.start > cursor + 0.05:
            kept.append((cursor, c.start))
        cursor = c.end
    if cursor < duration - 0.05:
        kept.append((cursor, duration))

    if not kept:
        print("WARNING: All content would be removed — aborting export.")
        video.close()
        return Summary()

    # --- Audio crossfade via pydub ---
    with tempfile.TemporaryDirectory() as tmp:
        # Extract full audio as wav
        full_audio_path = os.path.join(tmp, "full_audio.wav")
        video.audio.write_audiofile(full_audio_path, logger=None)
        full_audio = AudioSegment.from_wav(full_audio_path)

        fade_ms = int(crossfade_s * 1000)
        audio_segments: list[AudioSegment] = []

        for seg_start, seg_end in kept:
            start_ms = int(seg_start * 1000)
            end_ms = int(seg_end * 1000)
            audio_segments.append(full_audio[start_ms:end_ms])

        # Apply crossfades
        processed: list[AudioSegment] = []
        for idx, seg in enumerate(audio_segments):
            if idx < len(audio_segments) - 1:
                seg, audio_segments[idx + 1] = apply_audio_crossfade(
                    seg, audio_segments[idx + 1], fade_ms
                )
            processed.append(seg)

        combined_audio = processed[0]
        for seg in processed[1:]:
            combined_audio = combined_audio + seg

        combined_audio = normalize(combined_audio)
        crossfaded_audio_path = os.path.join(tmp, "crossfaded_audio.wav")
        combined_audio.export(crossfaded_audio_path, format="wav")

        # --- Video assembly via moviepy ---
        clips = []
        for seg_start, seg_end in kept:
            clip = video.subclip(seg_start, seg_end)
            clips.append(clip)

        final_video = concatenate_videoclips(clips, method="compose")

        # Replace audio with crossfaded version
        crossfaded_audio_clip = AudioFileClip(crossfaded_audio_path)
        # Trim audio to match video length (may differ by a few ms)
        audio_duration = min(crossfaded_audio_clip.duration, final_video.duration)
        crossfaded_audio_clip = crossfaded_audio_clip.subclip(0, audio_duration)
        final_video = final_video.set_audio(crossfaded_audio_clip)

        print(f"    Writing {output_path} …")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(tmp, "temp_audio_out.m4a"),
            remove_temp=True,
            logger=None,
            preset="slow",
            ffmpeg_params=["-crf", "18"],
        )

        final_video.close()
        video.close()

    # Build summary
    summary = Summary()
    for c in cuts:
        if not c.flagged:
            summary.filler_cuts += 1
            summary.time_saved += c.end - c.start
        else:
            summary.flagged.append(c)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_summary(summary: Summary, output_path: str) -> None:
    print("\n" + "=" * 60)
    print("  CLEAN VIDEO — SUMMARY")
    print("=" * 60)
    print(f"  Output file      : {output_path}")
    print(f"  Filler cuts made : {summary.filler_cuts}")
    print(f"  Total time saved : {summary.time_saved:.1f} seconds")

    if summary.flagged:
        print(f"\n  *** {len(summary.flagged)} cuts flagged for manual review ***")
        print("  These were SKIPPED (not removed) to avoid breaking sentences.\n")
        for i, c in enumerate(summary.flagged, 1):
            print(f"  [{i}] {c.start:.2f}s – {c.end:.2f}s  |  {c.reason}")
            print(f"       Reason: {c.flag_reason}")
    else:
        print("  No cuts flagged for manual review.")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove filler words from interview footage."
    )
    parser.add_argument("--input", required=True, help="Path to input MP4 or MOV file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base; use 'small' or 'medium' for better accuracy)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Seconds of audio to keep around each cut (default: 0.05)",
    )
    parser.add_argument(
        "--dump-transcript",
        action="store_true",
        help="Save the raw transcript with timestamps to a .json file",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    stem = Path(input_path).stem
    output_path = str(Path(input_path).parent / f"{stem}_cleaned.mp4")

    # 1. Transcribe
    words = transcribe(input_path, args.model)

    if args.dump_transcript:
        transcript_path = str(Path(input_path).parent / f"{stem}_transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(
                [{"text": w.text, "start": w.start, "end": w.end} for w in words],
                f, indent=2,
            )
        print(f"    Transcript saved to {transcript_path}")

    # 2. Detect fillers
    words = detect_fillers(words)

    # 3. Build cuts
    video = VideoFileClip(input_path)
    duration = video.duration
    video.close()
    cuts = build_cuts(words, duration)

    if not cuts:
        print("\nNo cuts to make — video is already clean!")
        sys.exit(0)

    # 4. Export
    summary = export_video(input_path, cuts, output_path)

    # 5. Print summary
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
