import random

import torch


def seed_everything(seed: int) -> None:
    """Set SEED for Python and PyTorch."""

    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
