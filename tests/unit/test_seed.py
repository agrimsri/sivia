"""Unit tests for seed_everything reproducibility."""

import random

from sivia.utils.seed import seed_everything


def test_seed_everything_determinism():
    """Verify that seed_everything produces deterministic random outputs."""
    seed_everything(12345)
    r1 = [random.random() for _ in range(5)]

    seed_everything(12345)
    r2 = [random.random() for _ in range(5)]

    assert r1 == r2

    # Verify NumPy determinism if available
    try:
        import numpy as np

        seed_everything(999)
        arr1 = np.random.rand(5)

        seed_everything(999)
        arr2 = np.random.rand(5)

        assert np.allclose(arr1, arr2)
    except ImportError:
        pass

    # Verify PyTorch determinism if available
    try:
        import torch

        seed_everything(42)
        t1 = torch.rand(4, 4)

        seed_everything(42)
        t2 = torch.rand(4, 4)

        assert torch.allclose(t1, t2)
    except ImportError:
        pass
