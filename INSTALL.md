# Setup & Install Instructions

## Prerequisites

- Python 3.9 or newer
- ffmpeg installed system-wide (required by moviepy and pydub)
- ImageMagick installed system-wide (required by moviepy TextClip for text overlays)

### Install ffmpeg + ImageMagick

**macOS (Homebrew):**
```bash
brew install ffmpeg imagemagick
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg imagemagick
```

**Windows:**
- ffmpeg: https://ffmpeg.org/download.html → add `bin/` to PATH
- ImageMagick: https://imagemagick.org/script/download.php

---

## Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

---

## Running the tool

### Filler removal only (no intro/outro)
```bash
python clean_video.py --input agent_interview.mp4
```

### With crossfade transitions between every cut
```bash
python clean_video.py --input agent_interview.mp4 --transition crossfade
```

### Full production run — intro card + lower third + outro + transitions
```bash
python clean_video.py --input agent_interview.mp4 \
  --model small \
  --transition crossfade \
  --intro-title "Meet Sarah Johnson" \
  --intro-subtitle "Top Agent · Miami, FL" \
  --intro-color 1a1a2e \
  --intro-duration 3 \
  --lower-third-name "Sarah Johnson" \
  --lower-third-title "Senior Real Estate Agent" \
  --lower-third-at 1.5 \
  --outro-text "Follow @BrowardRealty for more agent spotlights" \
  --outro-color 1a1a2e \
  --outro-duration 4
```

### With your agency logo on the intro card
```bash
python clean_video.py --input agent_interview.mp4 \
  --intro-title "Agent Spotlight" \
  --intro-logo path/to/logo.png \
  --intro-subtitle "BrowardRealty.com"
```

### Save the word-level transcript alongside the video
```bash
python clean_video.py --input agent_interview.mp4 --dump-transcript
```

---

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input MP4 or MOV path |
| `--model` | `base` | Whisper model: tiny / base / small / medium / large |
| `--transition` | `crossfade` | Transition between clips: `crossfade`, `fade_to_black`, `wipe_left`, `wipe_right`, `none` |
| `--intro-title` | — | Large title text on intro card |
| `--intro-subtitle` | — | Smaller subtitle on intro card |
| `--intro-logo` | — | Path to logo image (PNG with transparency works best) |
| `--intro-color` | `1a1a2e` | Hex background color for intro |
| `--intro-duration` | `3.0` | Intro card length in seconds |
| `--outro-text` | — | Call-to-action text on outro card |
| `--outro-color` | `1a1a2e` | Hex background color for outro |
| `--outro-duration` | `4.0` | Outro card length in seconds |
| `--lower-third-name` | — | Name displayed in lower-third bar |
| `--lower-third-title` | — | Job title in lower-third bar |
| `--lower-third-at` | `1.5` | Seconds into interview when bar appears |
| `--lower-third-duration` | `4.0` | How long the bar stays on screen |
| `--dump-transcript` | off | Save word-timestamp JSON alongside video |

---

## Output

| File | Description |
|------|-------------|
| `[name]_cleaned.mp4` | Final export — intro + clean interview + outro |
| `[name]_transcript.json` | Word timestamps (only with `--dump-transcript`) |

---

## Whisper model guide

| Model | Speed (CPU, 3-min clip) | Accuracy |
|-------|------------------------|----------|
| tiny  | ~1 min | Low |
| base  | ~2 min | Good |
| small | ~5 min | Better — recommended |
| medium | ~12 min | Great |
| large | ~25 min | Best |

---

## Manual review items

After the script finishes it prints flagged cuts — segments that were NOT removed
because removing them mid-sentence could break grammar. Open your editor, scrub to
each timestamp, and decide manually. Common cases:

- "like" used as a comparison: "it was like a 30-second call"
- "right" at end of a rhetorical question: "that's what buyers want, right?"
- "so" transitioning mid-idea: "so that means we're looking at…"
