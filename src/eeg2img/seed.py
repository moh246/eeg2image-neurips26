"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and optionally force deterministic kernels.

    ``deterministic=True`` trades some GPU speed for run-to-run repeatability.
    Data order is seeded separately through explicit ``torch.Generator`` and
    ``numpy.random.Generator`` objects, never through global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code may read the global state
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
