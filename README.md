# Responsible Gaming Shorts — MVP

work in progress project ingests **YouTube Shorts** (or uploaded .mp4), runs **ASR (whisper)** and **OCR (tesseract)**, and scores  content against a set of **Responsible Gaming** policy flags (i.e,  *risk-free*, *chasing losses*, *offshore brands*, *missing 1-800-GAMBLER*).

## Features
- Input: YouTube URL **or** upload .mp4
- pipe: ffmpeg → Whisper (ASR) → Tesseract OCR → rule-based flags → scoring
- Output: Overall risk score (0–100), category breakdown, flags, transcript, OCR text, representative frames.
- run in Codespaces via the incl. devcontainer

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

YouTube may say  **"Sign in to confirm you're not a bot"** unless `yt-dlp` uses your own authenticated cookies.

1. On your local machine, install a *cookies.txt* exporter in Chrome/Firefox and export cookies for `youtube.com` to a file 
2. Copy/upload file into the codespace 
3. In venv terminal before launching the app:

```bash
export YTDLP_COOKIES=/workspaces/RG_MVP/youtube_cookies.txt  # adjust path
unset YTDLP_BROWSER                                          # don't look for a browser profile
streamlit run app.py
```

Paste a **YouTube Shorts** URL or upload a short `.mp4`.

If the cookies expire and bot-check message returns, re-export and replace the file.

### Download clips without running the app

Use `download_clip.py` to fetch a YouTube video (or audio-only) as a local file. Supply the same cookie file or rely on `YTDLP_COOKIES`:

```bash
python download_clip.py "https://www.youtube.com/watch?v=abc123" --cookies /workspaces/RG_MVP/youtube_cookies.txt
# or, if YTDLP_COOKIES is already set:
python download_clip.py "https://www.youtube.com/watch?v=abc123" --audio-only
```


### GitHub Codespaces

```markdown
 repo includes a [devcontainer](.devcontainer) that installs `ffmpeg`, `tesseract-ocr`, and the Python requirements automatically. Run streamlit app thru codespaces or other text editor
```

## Dataset labeling and training **WORK IN PROGRESS**

Use  helper script to build a labeled transcript/OCR dataset:

```bash
python scripts/label_dataset.py path/to/video.mp4 dataset.jsonl
```

Tune multi label classifier on the collected data:

```bash
python models/transcript_classifier.py dataset.jsonl model_out_dir --epochs 3
```

## Notes
- This is a MVP with  semantic phrase matching and some logo detection, currently working on addig data to train the model via label_dataset.
- `operators.json` seeds offshore/sweepstakes/licensed names for detection
- `flags.py` holds regexes and embedding based phrase matchers; tune  as you observe false positives/negatives.
- For Reels, can potentially use the **Instagram Graph API** hashtag search to fetch public media metadata and (when available) `media_url`. 

## Current Proj Goals
- Train a DistilBERT text classifier on labeled clips
