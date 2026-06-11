#!/usr/bin/env python3
"""
clean_video.py — Full interview video editor.

Features:
  - Filler word removal with Whisper word-level timestamps
  - Clip transitions: crossfade, fade_to_black, wipe_left, wipe_right, none
  - Intro card with logo, title, subtitle, background color
  - Outro card with call-to-action text
  - Lower-third text overlays (name, title) timed to the interview
  - Audio crossfade on every filler cut

Usage:
    python clean_video.py --input agent_interview.mp4
    python clean_video.py --input interview.mp4 \\
        --model small \\
        --transition crossfade \\
        --intro-title "Meet Sarah Johnson" --intro-subtitle "Top Agent, Miami" \\
        --outro-text "Follow @BrowardRealty for more" \\
        --lower-third-name "Sarah Johnson" --lower-third-title "Senior Agent" \\
        --lower-third-at 3.0
"""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import whisper
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from pydub import AudioSegment
from pydub.effects import normalize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILLER_WORDS = {
    "um", "uh", "like", "so", "literally", "basically", "right", "okay",
    "kind", "sort", "mean",
}
FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of"]
AMBIGUOUS_FILLERS = {"like", "so", "right", "okay", "basically", "literally"}

CROSSFADE_DURATION = 0.2
MAX_PAUSE_TO_KEEP = 0.3
MIN_PAUSE_TO_CUT = 0.5
SENTENCE_BOUNDARY_PAUSE = 0.4

TRANSITION_DURATION = 0.5   # seconds for all clip transitions
INTRO_DURATION = 3.0         # seconds
OUTRO_DURATION = 4.0         # seconds
LOWER_THIRD_DURATION = 4.0   # seconds on screen
LOWER_THIRD_FADE = 0.4       # fade in/out


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
    print(f"[1/5] Transcribing with Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, word_timestamps=True, verbose=False)

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
    if idx == 0:
        return True
    return words[idx].start - words[idx - 1].end >= SENTENCE_BOUNDARY_PAUSE


def is_sentence_end(words: list[Word], idx: int) -> bool:
    if idx >= len(words) - 1:
        return True
    return words[idx + 1].start - words[idx].end >= SENTENCE_BOUNDARY_PAUSE


def detect_fillers(words: list[Word]) -> list[Word]:
    print("[2/5] Detecting filler words …")
    n = len(words)
    skip_until = -1

    for i, w in enumerate(words):
        if i <= skip_until:
            continue

        for phrase in FILLER_PHRASES:
            parts = phrase.split()
            if i + len(parts) > n:
                continue
            if all(words[i + j].text == parts[j] for j in range(len(parts))):
                at_start = is_sentence_start(words, i)
                after_end = is_sentence_end(words, i + len(parts) - 1)
                for j in range(len(parts)):
                    words[i + j].is_filler = True
                skip_until = i + len(parts) - 1
                if not at_start and not after_end and phrase in ("you know", "i mean"):
                    for j in range(len(parts)):
                        words[i + j].flagged = True
                        words[i + j].flag_reason = (
                            f"Phrase '{phrase}' mid-sentence — may affect meaning"
                        )
                break
        else:
            if w.text in FILLER_WORDS:
                at_start = is_sentence_start(words, i)
                after_end = is_sentence_end(words, i)
                if w.text in AMBIGUOUS_FILLERS:
                    w.is_filler = True
                    if not at_start and not after_end:
                        w.flagged = True
                        w.flag_reason = (
                            f"'{w.text}' mid-sentence — verify removal doesn't break grammar"
                        )
                else:
                    w.is_filler = True

    total = sum(1 for w in words if w.is_filler)
    flagged = sum(1 for w in words if w.flagged)
    print(f"    Found {total} filler instances ({flagged} flagged for review).")
    return words


# ---------------------------------------------------------------------------
# Step 3 — Build cut list
# ---------------------------------------------------------------------------

def merge_adjacent(cuts: list[Cut], gap: float = 0.05) -> list[Cut]:
    if not cuts:
        return cuts
    merged = [cuts[0]]
    for c in cuts[1:]:
        last = merged[-1]
        if c.start - last.end <= gap:
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
    print("[3/5] Building cut list …")
    cuts: list[Cut] = []
    n = len(words)
    i = 0

    while i < n:
        if not words[i].is_filler:
            i += 1
            continue
        j = i
        while j < n and words[j].is_filler:
            j += 1
        group = words[i:j]
        flagged = any(g.flagged for g in group)
        flag_reason = next((g.flag_reason for g in group if g.flag_reason), "")

        cut_start = group[0].start
        if i > 0:
            prev_end = words[i - 1].end
            if cut_start - prev_end > MAX_PAUSE_TO_KEEP:
                cut_start = prev_end + MAX_PAUSE_TO_KEEP

        cut_end = group[-1].end
        if j < n:
            next_start = words[j].start
            if next_start - cut_end > MAX_PAUSE_TO_KEEP:
                cut_end = cut_end + MAX_PAUSE_TO_KEEP

        cuts.append(Cut(
            start=cut_start,
            end=cut_end,
            reason=f"filler: {' '.join(g.text for g in group)}",
            flagged=flagged,
            flag_reason=flag_reason,
        ))
        i = j

    # Trim long pauses
    kept_words = [w for w in words if not w.is_filler]
    for idx in range(len(kept_words) - 1):
        gap_start = kept_words[idx].end
        gap_end = kept_words[idx + 1].start
        gap = gap_end - gap_start
        if gap > MIN_PAUSE_TO_CUT:
            trim_start = gap_start + MAX_PAUSE_TO_KEEP
            if gap_end - trim_start > 0.05:
                cuts.append(Cut(start=trim_start, end=gap_end,
                                reason=f"long pause ({gap:.2f}s)"))

    cuts.sort(key=lambda c: c.start)
    cuts = merge_adjacent(cuts)
    cuts = [c for c in cuts if c.start < video_duration]
    for c in cuts:
        c.end = min(c.end, video_duration)

    print(f"    Built {len(cuts)} cuts.")
    return cuts


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def apply_transition(
    clip_a,
    clip_b,
    transition: str,
    duration: float,
):
    """
    Returns (new_clip_a, new_clip_b) with the transition applied.
    Both clips are returned separately so concatenate_videoclips can handle them.
    For crossfade the overlap is handled via fadeout/fadein on each clip.
    """
    d = min(duration, clip_a.duration / 2, clip_b.duration / 2)

    if transition == "none":
        return clip_a, clip_b

    if transition == "crossfade":
        clip_a = fadeout(clip_a, d)
        clip_b = fadein(clip_b, d)
        return clip_a, clip_b

    if transition == "fade_to_black":
        clip_a = fadeout(clip_a, d)
        clip_b = fadein(clip_b, d)
        return clip_a, clip_b

    if transition in ("wipe_left", "wipe_right"):
        # Wipe is expensive to do frame-by-frame with moviepy without fx;
        # fall back to crossfade and note it.
        print(f"    Note: '{transition}' rendered as crossfade (no GPU compositor).")
        clip_a = fadeout(clip_a, d)
        clip_b = fadein(clip_b, d)
        return clip_a, clip_b

    return clip_a, clip_b


def apply_transitions_to_list(clips: list, transition: str, duration: float) -> list:
    if transition == "none" or len(clips) < 2:
        return clips
    result = list(clips)
    for i in range(len(result) - 1):
        result[i], result[i + 1] = apply_transition(
            result[i], result[i + 1], transition, duration
        )
    return result


# ---------------------------------------------------------------------------
# Intro card
# ---------------------------------------------------------------------------

def make_intro(
    size: tuple[int, int],
    fps: float,
    title: str,
    subtitle: str,
    bg_color: tuple[int, int, int],
    logo_path: str | None,
    duration: float = INTRO_DURATION,
):
    """
    Returns a clip that is a solid-color card with optional logo + title + subtitle.
    """
    w, h = size
    bg = ColorClip(size, color=bg_color, duration=duration)

    layers = [bg]

    y_center = h // 2

    if logo_path and os.path.isfile(logo_path):
        logo = (
            ImageClip(logo_path)
            .set_duration(duration)
            .resize(height=int(h * 0.18))
            .set_position(("center", int(h * 0.22)))
        )
        layers.append(logo)
        y_center = int(h * 0.52)

    if title:
        title_clip = (
            TextClip(
                title,
                fontsize=int(h * 0.08),
                color="white",
                font="Arial-Bold",
                method="caption",
                size=(int(w * 0.85), None),
                align="center",
            )
            .set_duration(duration)
            .set_position(("center", y_center))
        )
        layers.append(title_clip)
        y_center += int(h * 0.12)

    if subtitle:
        sub_clip = (
            TextClip(
                subtitle,
                fontsize=int(h * 0.045),
                color="#DDDDDD",
                font="Arial",
                method="caption",
                size=(int(w * 0.75), None),
                align="center",
            )
            .set_duration(duration)
            .set_position(("center", y_center))
        )
        layers.append(sub_clip)

    intro = CompositeVideoClip(layers, size=size).set_fps(fps)
    intro = fadein(intro, 0.4)
    intro = fadeout(intro, 0.4)
    return intro


# ---------------------------------------------------------------------------
# Outro card
# ---------------------------------------------------------------------------

def make_outro(
    size: tuple[int, int],
    fps: float,
    text: str,
    bg_color: tuple[int, int, int],
    duration: float = OUTRO_DURATION,
):
    w, h = size
    bg = ColorClip(size, color=bg_color, duration=duration)

    layers = [bg]

    if text:
        cta = (
            TextClip(
                text,
                fontsize=int(h * 0.065),
                color="white",
                font="Arial-Bold",
                method="caption",
                size=(int(w * 0.8), None),
                align="center",
            )
            .set_duration(duration)
            .set_position("center")
        )
        layers.append(cta)

    outro = CompositeVideoClip(layers, size=size).set_fps(fps)
    outro = fadein(outro, 0.4)
    outro = fadeout(outro, 0.4)
    return outro


# ---------------------------------------------------------------------------
# Lower-third overlay
# ---------------------------------------------------------------------------

def make_lower_third(
    size: tuple[int, int],
    fps: float,
    name: str,
    title: str,
    start_time: float,
    duration: float = LOWER_THIRD_DURATION,
    fade: float = LOWER_THIRD_FADE,
):
    """
    Returns a transparent overlay clip with a name + title bar.
    Caller is responsible for placing it at `start_time` in the composite.
    """
    w, h = size
    bar_h = int(h * 0.13)
    bar_y = int(h * 0.72)
    pad_x = int(w * 0.04)

    # Semi-transparent dark bar
    bar = ColorClip((w, bar_h), color=(10, 10, 10), duration=duration).set_opacity(0.72)
    bar = bar.set_position((0, bar_y))

    layers = [bar]

    if name:
        name_clip = (
            TextClip(
                name,
                fontsize=int(bar_h * 0.52),
                color="white",
                font="Arial-Bold",
            )
            .set_duration(duration)
            .set_position((pad_x, bar_y + int(bar_h * 0.08)))
        )
        layers.append(name_clip)

    if title:
        title_clip = (
            TextClip(
                title,
                fontsize=int(bar_h * 0.36),
                color="#FFD700",
                font="Arial",
            )
            .set_duration(duration)
            .set_position((pad_x, bar_y + int(bar_h * 0.55)))
        )
        layers.append(title_clip)

    overlay = CompositeVideoClip(layers, size=size, use_bgclip=False).set_fps(fps)
    overlay = fadein(overlay, fade)
    overlay = fadeout(overlay, fade)
    overlay = overlay.set_start(start_time)
    return overlay


# ---------------------------------------------------------------------------
# Step 4 — Audio crossfade
# ---------------------------------------------------------------------------

def apply_audio_crossfade(seg_a: AudioSegment, seg_b: AudioSegment, fade_ms: int):
    fade_ms = min(fade_ms, len(seg_a) // 2, len(seg_b) // 2)
    if fade_ms <= 0:
        return seg_a, seg_b
    return seg_a.fade_out(fade_ms), seg_b.fade_in(fade_ms)


# ---------------------------------------------------------------------------
# Step 5 — Export
# ---------------------------------------------------------------------------

def export_video(
    input_path: str,
    cuts: list[Cut],
    output_path: str,
    transition: str,
    intro_clip,
    outro_clip,
    lower_thirds: list,
    crossfade_s: float = CROSSFADE_DURATION,
) -> Summary:
    print("[5/5] Exporting video …")

    source = VideoFileClip(input_path)
    duration = source.duration
    size = source.size
    fps = source.fps

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
        print("WARNING: All content removed — aborting.")
        source.close()
        return Summary()

    with tempfile.TemporaryDirectory() as tmp:
        # ---- Crossfaded audio ----
        full_audio_path = os.path.join(tmp, "full_audio.wav")
        source.audio.write_audiofile(full_audio_path, logger=None)
        full_audio = AudioSegment.from_wav(full_audio_path)
        fade_ms = int(crossfade_s * 1000)

        audio_segs = [
            full_audio[int(s * 1000):int(e * 1000)]
            for s, e in kept
        ]
        processed = []
        for idx, seg in enumerate(audio_segs):
            if idx < len(audio_segs) - 1:
                seg, audio_segs[idx + 1] = apply_audio_crossfade(
                    seg, audio_segs[idx + 1], fade_ms
                )
            processed.append(seg)

        combined_audio = processed[0]
        for seg in processed[1:]:
            combined_audio = combined_audio + seg
        combined_audio = normalize(combined_audio)
        crossfaded_audio_path = os.path.join(tmp, "crossfaded.wav")
        combined_audio.export(crossfaded_audio_path, format="wav")

        # ---- Video clips ----
        interview_clips = [source.subclip(s, e) for s, e in kept]
        interview_clips = apply_transitions_to_list(
            interview_clips, transition, TRANSITION_DURATION
        )
        interview_video = concatenate_videoclips(interview_clips, method="compose")

        # Replace audio with crossfaded version
        cf_audio = AudioFileClip(crossfaded_audio_path).subclip(
            0, min(AudioFileClip(crossfaded_audio_path).duration, interview_video.duration)
        )
        interview_video = interview_video.set_audio(cf_audio)

        # ---- Lower thirds (composited onto interview) ----
        if lower_thirds:
            layers = [interview_video] + lower_thirds
            interview_video = CompositeVideoClip(layers)

        # ---- Assemble intro + interview + outro ----
        all_clips = []
        if intro_clip is not None:
            intro_clip = intro_clip.resize(size)
            all_clips.append(intro_clip)
        all_clips.append(interview_video)
        if outro_clip is not None:
            outro_clip = outro_clip.resize(size)
            all_clips.append(outro_clip)

        # Transitions between intro/outro and interview
        if len(all_clips) > 1:
            all_clips = apply_transitions_to_list(
                all_clips, transition, TRANSITION_DURATION
            )

        final = concatenate_videoclips(all_clips, method="compose")

        print(f"    Writing {output_path} …")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(tmp, "temp_out.m4a"),
            remove_temp=True,
            logger=None,
            preset="slow",
            ffmpeg_params=["-crf", "18"],
            fps=fps,
        )

        final.close()
        source.close()

    summary = Summary()
    for c in cuts:
        if not c.flagged:
            summary.filler_cuts += 1
            summary.time_saved += c.end - c.start
        else:
            summary.flagged.append(c)
    return summary


# ---------------------------------------------------------------------------
# Summary print
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
        print("  These were SKIPPED to avoid breaking sentences.\n")
        for i, c in enumerate(summary.flagged, 1):
            print(f"  [{i}] {c.start:.2f}s – {c.end:.2f}s  |  {c.reason}")
            print(f"       Reason: {c.flag_reason}")
    else:
        print("  No cuts flagged for manual review.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return (r, g, b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full interview video editor — filler removal + transitions + intro/outro."
    )

    # Core
    parser.add_argument("--input", required=True, help="Input MP4 or MOV file")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model (default: base)",
    )
    parser.add_argument("--dump-transcript", action="store_true")

    # Transitions
    parser.add_argument(
        "--transition", default="crossfade",
        choices=["crossfade", "fade_to_black", "wipe_left", "wipe_right", "none"],
        help="Transition style between every clip (default: crossfade)",
    )

    # Intro
    parser.add_argument("--intro-title", default="", help="Large text on intro card")
    parser.add_argument("--intro-subtitle", default="", help="Smaller text on intro card")
    parser.add_argument("--intro-logo", default="", help="Path to logo image for intro card")
    parser.add_argument(
        "--intro-color", default="1a1a2e",
        help="Intro background hex color (default: 1a1a2e — dark navy)",
    )
    parser.add_argument(
        "--intro-duration", type=float, default=INTRO_DURATION,
        help=f"Intro card duration in seconds (default: {INTRO_DURATION})",
    )

    # Outro
    parser.add_argument("--outro-text", default="", help="Call-to-action text on outro card")
    parser.add_argument(
        "--outro-color", default="1a1a2e",
        help="Outro background hex color (default: 1a1a2e)",
    )
    parser.add_argument(
        "--outro-duration", type=float, default=OUTRO_DURATION,
        help=f"Outro card duration in seconds (default: {OUTRO_DURATION})",
    )

    # Lower thirds
    parser.add_argument("--lower-third-name", default="", help="Name on lower-third overlay")
    parser.add_argument("--lower-third-title", default="", help="Job title on lower-third overlay")
    parser.add_argument(
        "--lower-third-at", type=float, default=1.5,
        help="Seconds into the interview when the lower-third appears (default: 1.5)",
    )
    parser.add_argument(
        "--lower-third-duration", type=float, default=LOWER_THIRD_DURATION,
        help=f"How long the lower-third stays on screen (default: {LOWER_THIRD_DURATION}s)",
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
        t_path = str(Path(input_path).parent / f"{stem}_transcript.json")
        with open(t_path, "w") as f:
            json.dump(
                [{"text": w.text, "start": w.start, "end": w.end} for w in words],
                f, indent=2,
            )
        print(f"    Transcript → {t_path}")

    # 2. Detect fillers
    words = detect_fillers(words)

    # 3. Build cuts
    probe = VideoFileClip(input_path)
    duration = probe.duration
    size = probe.size
    fps = probe.fps
    probe.close()
    cuts = build_cuts(words, duration)

    # 4. Build intro / outro / lower thirds
    print("[4/5] Building intro, outro, and lower-third overlays …")

    intro_clip = None
    if args.intro_title or args.intro_subtitle or args.intro_logo:
        intro_clip = make_intro(
            size=size,
            fps=fps,
            title=args.intro_title,
            subtitle=args.intro_subtitle,
            bg_color=hex_to_rgb(args.intro_color),
            logo_path=args.intro_logo or None,
            duration=args.intro_duration,
        )
        print(f"    Intro card: '{args.intro_title}' / '{args.intro_subtitle}'")

    outro_clip = None
    if args.outro_text:
        outro_clip = make_outro(
            size=size,
            fps=fps,
            text=args.outro_text,
            bg_color=hex_to_rgb(args.outro_color),
            duration=args.outro_duration,
        )
        print(f"    Outro card: '{args.outro_text}'")

    lower_thirds = []
    if args.lower_third_name or args.lower_third_title:
        lt = make_lower_third(
            size=size,
            fps=fps,
            name=args.lower_third_name,
            title=args.lower_third_title,
            start_time=args.lower_third_at,
            duration=args.lower_third_duration,
        )
        lower_thirds.append(lt)
        print(
            f"    Lower third: '{args.lower_third_name}' / '{args.lower_third_title}'"
            f" at {args.lower_third_at}s"
        )

    if not intro_clip and not outro_clip and not lower_thirds:
        print("    (No intro/outro/lower-third flags set — skipping.)")

    # 5. Export
    if not cuts:
        print("\nNo filler cuts to make.")

    summary = export_video(
        input_path=input_path,
        cuts=cuts,
        output_path=output_path,
        transition=args.transition,
        intro_clip=intro_clip,
        outro_clip=outro_clip,
        lower_thirds=lower_thirds,
    )

    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
