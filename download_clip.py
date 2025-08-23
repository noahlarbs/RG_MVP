#!/usr/bin/env python3
"""
download_clip.py — fetch MP4 (or audio) from a YouTube URL and print the saved path.

Usage:
  python download_clip.py "https://www.youtube.com/shorts/abc123"
  python download_clip.py "https://www.youtube.com/watch?v=abc123" -o ./downloads
  python download_clip.py "https://www.youtube.com/watch?v=abc123" --audio-only
  python download_clip.py "https://www.youtube.com/watch?v=abc123" --cookies /path/to/youtube_cookies.txt

Auth options (in order of precedence):
  --cookies /path/to/cookies.txt
  $YTDLP_COOKIES=/path/to/cookies.txt
  $YTDLP_BROWSER=chrome|safari|firefox|brave
"""

import argparse, os, sys, re, pathlib, subprocess
from typing import Optional
from yt_dlp import YoutubeDL

def normalize_youtube_url(url: str) -> str:
    # Convert Shorts → watch?v= format (more reliable)
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})", url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return url

def build_ydl_opts(outdir: pathlib.Path, cookies_arg: Optional[str], audio_only: bool) -> dict:
    # Prefer mp4+avc+aac (18) with fallbacks that avoid SABR-only streams
    format_sel = (
        "bestaudio[ext=m4a]/best"
        if audio_only else
        "18/(bv*[vcodec^=avc1]/b)[height<=720]+(ba[acodec^=mp4a]/b)/b"
    )
    opts = {
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "format": format_sel,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        # try multiple player clients to dodge some anti-bot variants
        "extractor_args": {"youtube": {"player_client": ["web", "ios", "android"]}},
        "http_headers": {"User-Agent": "Mozilla/5.0"},
    }

    # Auth/cookies precedence: CLI --cookies > env YTDLP_COOKIES > env YTDLP_BROWSER
    if cookies_arg:
        opts["cookiefile"] = cookies_arg
    else:
        env_cookie = os.environ.get("YTDLP_COOKIES")
        if env_cookie and os.path.exists(env_cookie):
            opts["cookiefile"] = env_cookie
        else:
            browser = os.environ.get("YTDLP_BROWSER")
            if browser:
                opts["cookiesfrombrowser"] = (browser, )

    return opts

def ensure_mp4(path: str) -> str:
    # Make sure final is .mp4 when not audio-only
    if path.lower().endswith(".mp4"):
        return path
    base, ext = os.path.splitext(path)
    mp4_path = base + ".mp4"
    # remux container without re-encoding
    subprocess.run(["ffmpeg", "-y", "-i", path, "-c", "copy", mp4_path], check=True)
    return mp4_path

def main():
    parser = argparse.ArgumentParser(description="Download a YouTube clip as MP4 (or audio-only) and print the saved path.")
    parser.add_argument("url", help="YouTube URL (Shorts or watch)")
    parser.add_argument("-o", "--out", default="./downloads", help="Output directory (default: ./downloads)")
    parser.add_argument("--cookies", help="Path to cookies.txt (overrides env)")
    parser.add_argument("--audio-only", action="store_true", help="Download audio-only (m4a best) instead of full video")
    args = parser.parse_args()

    outdir = pathlib.Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    url = normalize_youtube_url(args.url)
    ydl_opts = build_ydl_opts(outdir, args.cookies, args.audio_only)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            saved = ydl.prepare_filename(info)  # path with the actual extension

        if args.audio_only:
            # audio-only: just print the saved file (usually .m4a)
            final_path = saved
        else:
            # ensure we return an .mp4 (remux if needed)
            final_path = ensure_mp4(saved)

        print(final_path)
        return 0

    except Exception as e:
        sys.stderr.write(f"[download_clip] ERROR: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
