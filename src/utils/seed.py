import random

import numpy as np
import torch

from src.config.settings import ex

SEED: int = ex.configurations[0]()["project"]["seed"]


def seed_everything():
    """Set SEED for Python, NumPy, and PyTorch (CPU & GPU)."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    Each worker gets a deterministic SEED based on the global SEED + worker ID.
    """

    worker_seed = worker_id + SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
