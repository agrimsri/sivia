"""DINOv2 ViT-S/14 image embedding extraction module (Task M1.5).

Generates L2-normalized 384-dimensional CLS token embeddings for frame deduplication,
similarity search, and drift detection.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class DinoV2Embedder:
    """Extracts DINOv2 ViT-S/14 features using PyTorch Hub or Transformers."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
        use_half: bool = False,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.use_half = use_half and (self.device.type == "cuda")
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            if self.use_half:
                self._model.half()
        except Exception:
            # Fallback to torch.hub
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self._model.to(self.device)
            self._model.eval()
            if self.use_half:
                self._model.half()

    @torch.no_grad()
    def embed_batch(self, images: Sequence[np.ndarray | Image.Image | str | Path]) -> np.ndarray:
        """Embed a batch of images and return L2-normalized CLS token embeddings (N, 384)."""
        self._load_model()
        pil_images: list[Image.Image] = []

        for item in images:
            if isinstance(item, (str, Path)):
                img = Image.open(item).convert("RGB")
            elif isinstance(item, np.ndarray):
                # OpenCV BGR to RGB
                rgb = (
                    cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                    if len(item.shape) == 3 and item.shape[2] == 3
                    else item
                )
                img = Image.fromarray(rgb)
            elif isinstance(item, Image.Image):
                img = item.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(item)}")
            pil_images.append(img)

        if not pil_images:
            return np.empty((0, 384), dtype=np.float32)

        if self._processor is not None:
            inputs = self._processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.use_half:
                pixel_values = pixel_values.half()

            outputs = self._model(pixel_values=pixel_values)
            # CLS token embedding: shape (batch_size, 384)
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # Manual normalization for torch.hub model
            tensors = []
            for im in pil_images:
                im_resized = im.resize((224, 224), Image.Resampling.BICUBIC)
                arr = np.array(im_resized, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (arr - mean) / std
                tensors.append(torch.from_numpy(norm.transpose(2, 0, 1)))
            batch = torch.stack(tensors).to(self.device)
            if self.use_half:
                batch = batch.half()
            features = self._model(batch)

        # L2 Normalize
        norms = torch.norm(features, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        normalized = features / norms
        return normalized.cpu().float().numpy()


def save_embeddings(embeddings: np.ndarray, file_path: str | Path) -> Path:
    """Save embeddings array to .npy file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)
    return path


def load_embeddings(file_path: str | Path) -> np.ndarray:
    """Load embeddings array from .npy file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(str(path))
