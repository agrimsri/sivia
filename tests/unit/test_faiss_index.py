"""Unit tests for FAISS vector index."""

from pathlib import Path

import numpy as np

from sivia.embeddings.faiss_index import FaissIndex


def test_faiss_index_add_and_search(tmp_path: Path):
    """Test adding L2-normalized embeddings and searching nearest neighbors."""
    dim = 64
    index = FaissIndex(dimension=dim)
    assert index.size() == 0

    # 10 random normalized vectors
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((10, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    sample_ids = list(range(100, 110))
    index.add(embeddings, sample_ids)
    assert index.size() == 10

    # Query with vector 0 -> should match sample_id 100 with similarity close to 1.0
    query = embeddings[0:1]
    sims, matched_ids = index.search(query, top_k=3)

    assert matched_ids[0][0] == 100
    assert np.isclose(sims[0][0], 1.0, atol=1e-5)

    # Test save and reload
    save_path = tmp_path / "test.index"
    index.save(save_path)
    assert save_path.exists()

    loaded = FaissIndex.load(save_path, dimension=dim)
    assert loaded.size() == 10
    sims2, matched_ids2 = loaded.search(query, top_k=3)
    assert matched_ids2[0][0] == 100
