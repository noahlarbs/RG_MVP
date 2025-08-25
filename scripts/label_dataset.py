import argparse
import json
from pathlib import Path
from typing import Dict, List

from pipeline import process_video_file

LABELS = [
    "missing_disclaimer",
    "offshore_reference",
    "risk_free",
    "chasing_losses",
    "solve_financial_problems",
    "vpn_proxy",
]

def label_video(video: str, out_file: Path) -> None:
    """Process a video and prompt the user for RG flags.

    The transcript and OCR text are printed to the console so the annotator can
    make a decision without watching the full video.
    """
    result = process_video_file(video)
    transcript = result.get("transcript", "")
    ocr_text = result.get("ocr_text", "")

    print("\n=== Transcript ===\n", transcript)
    if ocr_text:
        print("\n=== OCR Text ===\n", ocr_text)

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
    args = p.parse_args()
    label_video(args.video, Path(args.out))


if __name__ == "__main__":
    main()
