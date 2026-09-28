"""SIVIA image embeddings and FAISS vector indexing package."""

from sivia.embeddings.dino import DinoV2Embedder, load_embeddings, save_embeddings
from sivia.embeddings.faiss_index import FaissIndex

__all__ = [
    "DinoV2Embedder",
    "save_embeddings",
    "load_embeddings",
    "FaissIndex",
]
