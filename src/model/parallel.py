"""
Data structures for parallel processing communication.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch.multiprocessing as mp


@dataclass
class Task:
    """Represents a single unit of work: evaluating one individual."""

    index: int
    neural_network_config: Dict[str, Any]
    individual_hyperparams: Dict[str, Any]
    training_epochs: int
    early_stop_epochs: int
    subset_percentage: float
    pop_size: int
    train_indices: Optional[np.ndarray]
    test_indices: Optional[np.ndarray]
    is_final: bool = False


@dataclass
class Result:
    """Represents the outcome of a completed task."""

    index: int
    fitness: float
    loss: float
    status: str  # 'SUCCESS', 'FAILURE', 'CANCELLED'
    log_lines: [str]
    duration_seconds: float
    error_message: Optional[str] = None


@dataclass
class WorkerConfig:
    """Static configuration passed to each worker process on startup."""

    worker_id: int
    device: Union[str, int]  # 'cpu' or GPU index (0, 1, ...)
    task_queue: mp.Queue
    result_queue: mp.Queue
    session_log_filename: str
    num_dataloader_workers: int
