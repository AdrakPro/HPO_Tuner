import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set SEED for Python, NumPy, and PyTorch (CPU & GPU)."""

    if seed is None:
        return

    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
