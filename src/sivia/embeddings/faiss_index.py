"""FAISS vector indexing and nearest-neighbor search module (Task M1.7).

Maintains a persistent cosine-similarity FAISS index mapped directly to SQLite sample IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """Persistent cosine-similarity FAISS index with integer ID mapping."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        # Inner product index for L2-normalized vectors equals cosine similarity
        raw_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(raw_index)

    def add(self, embeddings: np.ndarray, ids: Sequence[int]) -> None:
        """Add batch of normalized embeddings with their associated database IDs."""
        if len(embeddings) == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        ids_arr = np.array(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add_with_ids(vectors, ids_arr)

    def search(self, queries: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search top_k nearest neighbors. Returns (similarities, matched_ids)."""
        vectors = np.ascontiguousarray(queries, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        if self.index.ntotal == 0:
            empty_sims = np.empty((len(vectors), 0), dtype=np.float32)
            empty_ids = np.empty((len(vectors), 0), dtype=np.int64)
            return empty_sims, empty_ids

        k = min(top_k, self.index.ntotal)
        similarities, matched_ids = self.index.search(vectors, k)
        return similarities, matched_ids

    def size(self) -> int:
        return int(self.index.ntotal)

    def save(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        return path

    @classmethod
    def load(cls, file_path: str | Path, dimension: int = 384) -> FaissIndex:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        instance = cls(dimension=dimension)
        instance.index = faiss.read_index(str(path))
        return instance
