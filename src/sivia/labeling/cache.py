"""Parquet-backed cache for zero-shot teacher model predictions.

Caches raw detector outputs keyed by (image_sha256, model_id, prompt_fingerprint)
to ensure deterministic reproducibility and prevent redundant GPU/CPU passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sivia.labeling.types import RawDetection


class DetectionCache:
    """Parquet disk cache for object detection inferences."""

    def __init__(self, cache_dir: str | Path = "data/cache/detections") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        image_sha256: str,
        model_id: str,
        prompts: dict[int, str] | list[str] | dict[str, Any],
    ) -> str:
        """Compute deterministic SHA256 cache key."""
        serialized_prompts = json.dumps(prompts, sort_keys=True)
        raw_key = f"{image_sha256}_{model_id}_{serialized_prompts}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> list[RawDetection] | None:
        """Retrieve cached detections if available, otherwise None."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)
            detections: list[RawDetection] = []
            for _, row in df.iterrows():
                detections.append(
                    RawDetection(
                        class_id=int(row["class_id"]),
                        class_name=str(row["class_name"]),
                        box_xyxy=[
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ],
                        score=float(row["score"]),
                        prompt=str(row.get("prompt", "")),
                        model_name=str(row.get("model_name", "")),
                    )
                )
            return detections
        except Exception:
            return None

    def put(self, cache_key: str, detections: list[RawDetection]) -> Path:
        """Save detections to Parquet cache."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        records = []
        for det in detections:
            records.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "x1": det.box_xyxy[0],
                    "y1": det.box_xyxy[1],
                    "x2": det.box_xyxy[2],
                    "y2": det.box_xyxy[3],
                    "score": det.score,
                    "prompt": det.prompt,
                    "model_name": det.model_name,
                }
            )

        if not records:
            # Write empty dataframe with schema
            df = pd.DataFrame(
                columns=[
                    "class_id",
                    "class_name",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "score",
                    "prompt",
                    "model_name",
                ]
            )
        else:
            df = pd.DataFrame(records)

        df.to_parquet(cache_file, index=False)
        return cache_file
