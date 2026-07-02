"""Reproducibility helpers."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Seed all RNGs for reproducible runs. When ``deterministic`` is set, also
    force cuDNN into deterministic mode (slower but bit-reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed
