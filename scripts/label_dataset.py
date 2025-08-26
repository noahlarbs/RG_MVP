import argparse
import json
from pathlib import Path
from typing import Dict, List

from pipeline import process_video_file
from scorer import WEIGHTS

# Use flags defined in scorer to keep labeling options in sync
LABELS = sorted(WEIGHTS.keys())

def _summarize_ocr(text: str, operators: List[str]) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keywords = [op.lower() for op in operators] + ["21", "gambl", "stake", "rainbet", "bovada", "roobet"]
    key_lines = [ln for ln in lines if any(k in ln.lower() for k in keywords)]
    return "\n".join(key_lines[:10])


def label_video(video: str, out_file: Path, fast: bool = False) -> None:
    """Process a video and prompt the user for RG flags.

    The transcript and OCR text are printed to the console so the annotator can
    make a decision without watching the full video.
    """
    result = process_video_file(video, use_embed=not fast, use_logos=not fast)
    transcript = result.get("transcript", "")
    ocr_text = result.get("ocr_text", "")
    operators = result.get("features", {}).get("operators", [])
    hits = result.get("hits", {})

    print("\n=== Transcript ===\n", transcript)
    if ocr_text:
        summary = _summarize_ocr(ocr_text, operators)
        if summary:
            print("\n=== OCR Snippets ===\n", summary)
    if operators:
        print("\nDetected operators:", ", ".join(sorted(operators)))
    if hits:
        print("Detected phrases:", ", ".join(sorted(hits.keys())))

    print("\nEnter comma-separated flags from:\n" + ", ".join(LABELS))
    raw = input("flags> ").strip()
    flagged = {lbl: False for lbl in LABELS}
    if raw:
        for lbl in [f.strip() for f in raw.split(",") if f.strip()]:
            if lbl in flagged:
                flagged[lbl] = True

    entry: Dict[str, object] = {
        "video": video,
        "transcript": transcript,
        "ocr_text": ocr_text,
        "flags": flagged,
    }
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Saved labels to {out_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Label a video for RG violations")
    p.add_argument("video", help="Path to mp4 file")
    p.add_argument("out", help="Path to output JSONL file")
    p.add_argument("--fast", action="store_true", help="Skip heavy embed and logo models for speed")
    args = p.parse_args()
    label_video(args.video, Path(args.out), fast=args.fast)


if __name__ == "__main__":
    main()
