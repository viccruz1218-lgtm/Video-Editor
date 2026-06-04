# Setup & Install Instructions

## Prerequisites

- Python 3.9 or newer
- ffmpeg installed system-wide (required by both moviepy and pydub)

### Install ffmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add the `bin/` folder to your PATH.

---

## Python dependencies

Create and activate a virtual environment first (strongly recommended):

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows
```

Install all Python packages:

```bash
pip install -r requirements.txt
```

> **Note on Torch:** `openai-whisper` depends on PyTorch. The command above installs
> the CPU version. If you have an NVIDIA GPU and want faster transcription, install
> the GPU build of torch first — see https://pytorch.org/get-started/locally/

---

## Running the tool

Basic usage (uses the `base` Whisper model — fast, good enough for clear speech):

```bash
python clean_video.py --input agent_interview.mp4
```

Better accuracy (slower — recommended for noisy audio or strong accents):

```bash
python clean_video.py --input agent_interview.mp4 --model small
```

Best accuracy (slowest — use on a GPU machine or overnight):

```bash
python clean_video.py --input agent_interview.mp4 --model medium
```

Save the raw transcript with word-level timestamps alongside the video:

```bash
python clean_video.py --input agent_interview.mp4 --dump-transcript
```

---

## Output

| File | Description |
|------|-------------|
| `agent_interview_cleaned.mp4` | Final export — same resolution/quality as input |
| `agent_interview_transcript.json` | Word-level timestamps (only with `--dump-transcript`) |

---

## Whisper model size guide

| Model | VRAM | Speed (CPU) | Accuracy |
|-------|------|-------------|----------|
| tiny  | ~1 GB | Very fast | Low |
| base  | ~1 GB | Fast | Good |
| small | ~2 GB | Moderate | Better |
| medium | ~5 GB | Slow | Great |
| large | ~10 GB | Very slow | Best |

For 2–5 minute interviews on a modern MacBook, `base` takes ~2 min and `small` takes ~5 min.

---

## Manual review items

After the script finishes it prints a list of **flagged cuts** — segments it detected
as fillers but did NOT remove because they were mid-sentence in a context where removal
might break grammar. Open your video editor, scrub to each timestamp shown, and decide
whether to cut manually.

Common cases that get flagged:
- "like" used as a comparison ("it was like a 30-second call")
- "right" at end of a rhetorical question ("that's what buyers want, right?")
- "so" transitioning into a new idea mid-sentence
