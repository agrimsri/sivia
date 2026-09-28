"""Global reproducibility seeding utilities for SIVIA.

Configures random number generators across Python stdlib, NumPy, PyTorch,
and sets deterministic flags on cuDNN.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """Set seeds across Python, NumPy, PyTorch, and configure cuDNN determinism.

    Args:
        seed: The integer seed to use.
        deterministic_cudnn: Whether to enforce deterministic cuDNN convolution algorithms.

    Returns:
        The seed value that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn and hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
