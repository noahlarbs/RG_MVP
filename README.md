# Responsible Gaming Shorts — MVP

A fast, local-first prototype that ingests **YouTube Shorts** (or uploaded .mp4), runs **ASR (Whisper)** and **OCR (Tesseract)**, and scores the content against a policy-grounded set of **Responsible Gaming** flags (e.g., *risk-free*, *chasing losses*, *offshore brands*, *missing 1-800-GAMBLER*).

## Features
- Input: YouTube URL **or** upload .mp4
- Pipeline: ffmpeg → Whisper (ASR) → Tesseract OCR → rule-based flags → scoring
- Output: Overall risk score (0–100), category breakdown, flags, transcript, OCR text, representative frames.
- Ready-to-run in GitHub Codespaces via the included devcontainer

## Install

```bash
# System deps

# Mac: brew install ffmpeg tesseract
# Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg tesseract-ocr


python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Whisper will download its model on first run.


## Run

YouTube may respond with **"Sign in to confirm you're not a bot"** unless `yt-dlp` uses your own authenticated cookies.

1. On your local machine, install a *cookies.txt* exporter in Chrome/Firefox and export cookies for `youtube.com` to a file (e.g., `youtube_cookies.txt`).
2. Copy this file into the codespace (VS Code → *Explorer* → **Upload**…).
3. In your venv terminal before launching the app:

```bash
export YTDLP_COOKIES=/workspaces/RG_MVP/youtube_cookies.txt  # adjust path
unset YTDLP_BROWSER                                          # don't look for a browser profile
streamlit run app.py
```

Paste a **YouTube Shorts** URL or upload a short `.mp4`.

If the cookies expire and the bot-check message returns, re-export and replace the file.

### Download clips without running the app

Use `download_clip.py` to fetch a YouTube video (or audio-only) as a local file. Supply the same cookie file or rely on `YTDLP_COOKIES`:

```bash
python download_clip.py "https://www.youtube.com/watch?v=abc123" --cookies /workspaces/RG_MVP/youtube_cookies.txt
# or, if YTDLP_COOKIES is already set:
python download_clip.py "https://www.youtube.com/watch?v=abc123" --audio-only
```


### GitHub Codespaces

```markdown
This repo includes a [devcontainer](.devcontainer) that installs `ffmpeg`, `tesseract-ocr`, and the Python requirements automatically. Open in Codespaces and you're ready to run `streamlit`.
```

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
