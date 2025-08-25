"""Lightweight logo detection using CLIP embeddings.

Images of known offshore-operator logos should be placed under
`assets/logos/`. Detection computes CLIP embeddings for each frame and each
logo and reports any logo whose cosine similarity exceeds a threshold.
"""
from pathlib import Path
from typing import Iterable, Set

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class LogoDetector:
    def __init__(self, logo_dir: Path | None = None) -> None:
        self.logo_dir = Path(logo_dir or Path(__file__).parent / "assets/logos")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.logo_embeddings = {}
        self._load_logos()

    def _load_logos(self) -> None:
        for path in self.logo_dir.glob("*.png"):
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2)
            self.logo_embeddings[path.stem.lower()] = emb

    def detect(self, frame_paths: Iterable[str], threshold: float = 0.3) -> Set[str]:
        hits: Set[str] = set()
        if not self.logo_embeddings:
            return hits
        for fp in frame_paths:
            image = Image.open(fp).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2)
            for name, logo_emb in self.logo_embeddings.items():
                sim = torch.cosine_similarity(emb, logo_emb).item()
                if sim >= threshold:
                    hits.add(name)
        return hits
