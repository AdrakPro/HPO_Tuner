"""
Factory function for creating the appropriate population evaluator.
"""

from typing import Dict, Optional

import numpy as np
from rich.progress import TaskID

from src.genetic.individual_evaluator import IndividualEvaluator
from src.genetic.scheduling_strategy import (
    CPUOnlyStrategy,
    GPUOnlyStrategy,
    HybridStrategy,
    SchedulingStrategy,
)
from src.logger.logger import logger
from src.model.evaluator_interface import Evaluator
from src.parallel.parallel_evaluator import ParallelEvaluator
from src.tui.tui_screen import TUI


def create_evaluator(
    config: Dict,
    training_epochs: int,
    early_stop_epochs: int,
    subset_percentage: float,
    tui: TUI,
    task_id: TaskID,
    session_log_filename: str,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Evaluator:
    """
    Selects and creates the appropriate evaluator (sequential or parallel)
    based on the application configuration.
    """
    parallel_enabled = config["parallel_config"]["execution"]["enable_parallel"]
    neural_config = config["neural_network_config"]

    if not parallel_enabled:
        return IndividualEvaluator(
            neural_config=neural_config,
            training_epochs=training_epochs,
            early_stop_epochs=early_stop_epochs,
            subset_percentage=subset_percentage,
            progress=tui.progress,
            task_id=task_id,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    eval_mode = config["parallel_config"]["execution"]["evaluation_mode"]
    strategy: SchedulingStrategy

    if eval_mode == "CPU":
        strategy = CPUOnlyStrategy()
    elif eval_mode == "GPU":
        strategy = GPUOnlyStrategy()
    elif eval_mode == "HYBRID":
        strategy = HybridStrategy()
    else:
        logger.error(
            f"Unknown evaluation mode: {eval_mode}. Defaulting to sequential."
        )
        return IndividualEvaluator(
            neural_config=neural_config,
            training_epochs=training_epochs,
            early_stop_epochs=early_stop_epochs,
            subset_percentage=subset_percentage,
            progress=tui.progress,
            task_id=task_id,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    return ParallelEvaluator(
        config=config,
        training_epochs=training_epochs,
        early_stop_epochs=early_stop_epochs,
        subset_percentage=subset_percentage,
        strategy=strategy,
        progress=tui.progress,
        task_id=task_id,
        session_log_filename=session_log_filename,
        train_indices=train_indices,
        test_indices=test_indices,
    )
