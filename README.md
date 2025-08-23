# Responsible Gaming Shorts — MVP

A fast, local-first prototype that ingests **YouTube Shorts** (or uploaded .mp4), runs **ASR (Whisper)** and **OCR (PaddleOCR)**, and scores the content against a policy-grounded set of **Responsible Gaming** flags (e.g., *risk-free*, *chasing losses*, *offshore brands*, *missing 1-800-GAMBLER*).

## Features
- Input: YouTube URL **or** upload .mp4
- Pipeline: ffmpeg → Whisper (ASR) → PaddleOCR → rule-based flags → scoring
- Output: Overall risk score (0–100), category breakdown, flags, transcript, OCR text, representative frames.

## Install

```bash
# System deps
# Mac: brew install ffmpeg
# Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Whisper/Paddle will download models on first run.

## Run

1. In Chrome, install a "cookies.txt" exporter and export cookies for youtube.com to `~/youtube_cookies.txt`.
2. In your venv terminal before launching the app:

```bash
export YTDLP_COOKIES=~/youtube_cookies.txt
unset YTDLP_BROWSER
streamlit run app.py
```

Then paste a **YouTube Shorts** URL, or upload a short `.mp4`.

### Download clips without running the app

Use `download_clip.py` to fetch a YouTube video (or audio-only) as a local file:

```bash
python download_clip.py "https://www.youtube.com/watch?v=abc123"
python download_clip.py "https://www.youtube.com/watch?v=abc123" --audio-only
```

### GitHub Codespaces

This repo includes a [devcontainer](.devcontainer) that installs `ffmpeg` and the Python requirements automatically. Open in Codespaces and you're ready to run `streamlit`.

## Notes
- This is a **rules-first** MVP. You can later fine-tune a small transformer on labeled transcripts+OCR.
- `operators.json` seeds offshore/sweepstakes/licensed names for detection.
- `flags.py` holds the regexes; tune them as you observe false positives/negatives.
- For Instagram Reels, use the **Instagram Graph API** hashtag search to fetch public media metadata and (when available) `media_url`. This demo focuses on YouTube for speed.

## Roadmap
- Evidence PDF export with screenshots + policy citations
- CLIP logo matching and in-car scene classifier
- Instagram ingest via Graph API
- Train a DistilBERT text classifier on labeled clips
