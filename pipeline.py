import os, json, subprocess, tempfile, shutil, glob
from typing import Dict, Any, Tuple, List, Set
from dataclasses import dataclass
from pathlib import Path

import regex as re
import pytesseract
from PIL import Image
import whisper
from yt_dlp import YoutubeDL
# imports for flags
from flags import find_hits, PATTERNS, fuzzy_hits, embedding_hits
from scorer import score_clip
# imports for OCR and detection
import uuid
import shutil
import cv2
import numpy as np
from logo_detector import LogoDetector


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# Load operator registry
with open(os.path.join(os.path.dirname(__file__), "operators.json"), "r") as f:
    OPERATORS = json.load(f)

# Initialize once (lazy in real app)
_ASR = None
_LOGO = None

def _get_asr():
    global _ASR
    if _ASR is None:
        # try "small" if your machine can handle; else keep "base"
        _ASR = whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))
    return _ASR


def _get_logo_detector():
    global _LOGO
    if _LOGO is None:
        try:
            _LOGO = LogoDetector()
        except Exception:
            _LOGO = False  # sentinel for unavailable
    return _LOGO or None


def download_youtube(url: str, out_dir: str) -> str:
    # Normalize Shorts url to watch?v= form (more reliable)
    if "youtube.com/shorts/" in url:
        vid = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
        url = f"https://www.youtube.com/watch?v={vid}"

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        # Prefer mp4+avc+aac (format 18) and fallbacks that avoid SABR-only streams
        "format": "18/(bv*[vcodec^=avc1]/b)[height<=720]+(ba[acodec^=mp4a]/b)/b",
        "quiet": True,
        "noplaylist": True,
        # try multiple player clients to avoid signature issues / SABR
        "extractor_args": {"youtube": {"player_client": ["web", "ios", "android"]}},
        "http_headers": {"User-Agent": "Mozilla/5.0"},
    }


    # Use your browser session if available (recommended)
    # export YTDLP_BROWSER=chrome   (or safari/firefox/brave)
    browser = os.environ.get("YTDLP_BROWSER")
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)

    # Or a cookies.txt file (avoids repeated Keychain prompts)
    # export YTDLP_COOKIES=/path/to/youtube_cookies.txt
    cookiefile = os.environ.get("YTDLP_COOKIES")
    if cookiefile and os.path.exists(cookiefile):
        ydl_opts["cookiefile"] = cookiefile

    from yt_dlp import YoutubeDL

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        if not path.endswith(".mp4"):
            base = os.path.splitext(path)[0]
            mp4_path = base + ".mp4"
            subprocess.run(["ffmpeg","-y","-i", path, "-c","copy", mp4_path], check=True)
            path = mp4_path
        return path

def extract_audio(video_path: str, out_dir: str) -> str:
    audio_path = os.path.join(out_dir, "audio.wav")
    subprocess.run(["ffmpeg","-y","-i", video_path, "-ar","16000","-ac","1", audio_path], check=True)
    return audio_path

def extract_frames(video_path: str, out_dir: str, fps: int = 1) -> List[str]:
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(["ffmpeg","-y","-i", video_path, "-vf", f"fps={fps}", os.path.join(frames_dir, "frame_%05d.jpg")], check=True)
    return sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
#gets decoding options
def run_asr(audio_path: str) -> str:
    asr = _get_asr()
    result = asr.transcribe(
        audio_path,
        language="en",
        fp16=False,
        temperature=0.0,
        no_speech_threshold=0.25,
        logprob_threshold=-1.0,
        best_of=5,
        beam_size=5,
    )
    return result.get("text","").strip()


def _ocr_one_image(pil_img: Image.Image) -> str:
    # Try multiple pre-process pipelines; keep best text
    img = np.array(pil_img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    texts = []

    def ocr_with(img_gray, psm):
        # Upscale for small caption fonts
        scaled = cv2.resize(img_gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        # Contrast normalize
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm = clahe.apply(scaled)
        # Two binarizations
        th1 = cv2.adaptiveThreshold(norm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,11)
        th2 = cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cfg = f'--oem 3 --psm {psm}'
        for mat in (norm, th1, th2):
            data = pytesseract.image_to_data(mat, output_type=pytesseract.Output.DICT, config=cfg)
            words = [w for w, conf in zip(data["text"], data["conf"]) if w and str(conf).isdigit() and int(conf) >= 60]
            if words:
                texts.append(" ".join(words))

    # Whole image
    for psm in (6, 7, 11):  # block, single line, sparse text
        ocr_with(gray, psm)

    # Crop likely caption regions (top/bottom thirds)
    h = gray.shape[0]
    top = gray[0:int(0.28*h), :]
    mid = gray[int(0.36*h):int(0.64*h), :]
    bot = gray[int(0.72*h):, :]
    for region in (top, mid, bot):
        for psm in (6, 7, 11):
            ocr_with(region, psm)

    # Return the best concatenation
    return "\n".join(texts)

def run_ocr_on_frames(frames):
    texts = []
    # Bias sampling toward the first 10 seconds plus mid/last frames
    sample_ids = set()
    for i, f in enumerate(frames):
        if i < 10 or i in (len(frames)//2, len(frames)-1):
            sample_ids.add(i)
    for i in sorted(sample_ids):
        try:
            txt = _ocr_one_image(Image.open(frames[i]))
            if txt:
                texts.append(txt)
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract OCR executable not found. Install 'tesseract-ocr' to enable OCR processing"
            ) from e
        except Exception:
            pass
    return "\n".join(texts)


def normalize(text: str) -> str:
    # light normalization for regex
    return re.sub(r"\s+", " ", (text or "")).strip()

def detect_operators(text: str) -> Set[str]:
    found = set()
    low = text.lower()
    for name, meta in OPERATORS.items():
        aliases = set(meta.get("aliases", [])) | {name}
        for a in aliases:
            if a.lower() in low:
                found.add(name)
                break
    return found


def build_features(
    transcript: str,
    ocr_text: str,
    metadata: Dict[str, Any] = None,
    logos: Set[str] | None = None,
    use_embed: bool = True,
) -> Dict[str, Any]:

    transcript_n = normalize(transcript)
    ocr_n = normalize(ocr_text)
    joined = f"{transcript_n}\n{ocr_n}"

    hits = find_hits(joined)
    # Fuzzy phrase backstop (handles ASR/OCR imperfections)
    fuzzy = set()
    try:
        fuzzy = fuzzy_hits(joined, threshold=82)  # a bit lenient for Shorts
    except Exception:
        pass
    emb = set()

    if use_embed:
        try:
            emb = embedding_hits(joined)
        except Exception:
            pass


    phrases = set(hits.keys()) | fuzzy | emb

    operators = detect_operators(joined)
    if logos:
        operators |= logos

    features = {
        "phrases": phrases,
        "operators": list(operators),
        "has_helpline": bool(hits.get("helpline")),
        "has_21plus": bool(hits.get("age21")),
        "has_promo_terms": bool(hits.get("promo_terms")),
        "youth_context": bool(hits.get("youth_context")),
        "college_cues": bool(hits.get("college_cues")),
        "danger_driving": bool(hits.get("driving")),
        "socially_irresponsible": bool(hits.get("danger_social")),
        "vpn_proxy": bool(hits.get("vpn_proxy")),
        # Undisclosed affiliate: saw promo code but not #ad/#sponsored in transcript/ocr/meta
        "affiliate_undisclosed": ("promo" in phrases) and not re.search(r"#(ad|sponsored)\b", joined, re.I),
        # YouTube unapproved ref heuristic (placeholder): operator is offshore and promo mentioned
        "unapproved_ref": (len(operators & {"bovada","stake","roobet","rainbet","rollbit"})>0) and ("promo" in phrases),
    }
    return features, hits

def process_video_file(video_path: str, use_embed: bool = True, use_logos: bool = True) -> Dict[str,Any]:
    with tempfile.TemporaryDirectory() as tdir:
        # copy to temp
        temp_video = os.path.join(tdir, "video.mp4")
        shutil.copy(video_path, temp_video)
        audio = extract_audio(temp_video, tdir)
        frames = extract_frames(temp_video, tdir, fps=1)
        transcript = run_asr(audio)
        ocr_text = run_ocr_on_frames(frames)
        logos = set()

        if use_logos:
            detector = _get_logo_detector()
            if detector:
                try:
                    logos = detector.detect(frames)
                except Exception:
                    logos = set()
        features, hits = build_features(transcript, ocr_text, {}, logos=logos if use_logos else None, use_embed=use_embed)

        overall, cats, flags = score_clip(features)

        # pick representative frames (first, middle, last)
        reps = []
        if frames:
            cand = [frames[0]]
            if len(frames) >= 3:
                cand = [frames[0], frames[len(frames)//2], frames[-1]]

            # persist copies in ./artifacts so Streamlit can render after temp cleanup
            rep_paths = []
            run_id = uuid.uuid4().hex[:8]
            out_dir = ARTIFACTS_DIR / f"run_{run_id}"
            out_dir.mkdir(exist_ok=True)
            for idx, src in enumerate(cand):
                dst = out_dir / f"rep_{idx+1}.jpg"
                shutil.copyfile(src, dst)
                rep_paths.append(str(dst))
            reps = rep_paths

        return {
            "overall": overall,
            "categories": cats,
            "flags": flags,
            "features": features,
            "hits": hits,
            "transcript": transcript,
            "ocr_text": ocr_text,
            "logos": list(logos),
            "rep_frames": reps,
        }

def process_youtube(url: str) -> Dict[str,Any]:
    with tempfile.TemporaryDirectory() as tdir:
        path = download_youtube(url, tdir)
        return process_video_file(path)
