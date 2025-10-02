"""
Data structures for parallel execution.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
import multiprocessing as mp


@dataclass
class WorkerConfig:
    """Configuration for a worker process."""

    worker_id: int
    device: Union[int, str]  # GPU index or "cpu"
    task_queue: mp.Queue
    result_queue: mp.Queue
    session_log_filename: str
    num_dataloader_workers: int
    total_cpu_workers: Optional[int] = None
    fixed_batch_size: Optional[int] = None


@dataclass
class Task:
    """A task to be executed by a worker."""

    index: int
    neural_network_config: Dict[str, Any]
    individual_hyperparams: Dict[str, Any]
    training_epochs: int
    early_stop_epochs: int
    subset_percentage: float
    pop_size: int
    is_final: bool
    train_indices: Optional[List[int]]
    test_indices: Optional[List[int]]


@dataclass
class Result:
    """Result of a task execution."""

    index: int
    fitness: float
    loss: float
    duration_seconds: float
    status: str  # "SUCCESS", "FAILURE", "CANCELLED"
    log_lines: List[str | Tuple[str, str]]
    error_message: Optional[str] = None
