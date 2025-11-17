import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.logger.logger import logger


@dataclass
class GaState:
    """
    A dataclass representing the complete state of a GA run at a specific point.
    """

    generation: int
    population: List[Dict]
    fitness_scores: List[float]
    phase: str
    config: Dict[str, Any]
    session_log_filename: str
    phase_completed: bool
    outer_fold_k: int = -1
    rng_state: Optional[Dict[str, Any]] = None


class _CheckpointManager:
    """
    Manages saving and loading of the application state for resuming interrupted runs.
    """

    def __init__(
        self,
    ):
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        self.checkpoint_dir = f"/lustre/pd01/hpc-adamak7184-1759856296/checkpoints/{task_id}"
        self.filepath = os.path.join(self.checkpoint_dir, "ga_checkpoint.pkl")
        self.temp_filepath = f"{self.filepath}.tmp"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def is_checkpoint_exists(self) -> bool:
        """Checks if a valid checkpoint file exists."""
        return os.path.exists(self.filepath)

    def save_checkpoint(self, state: GaState) -> None:
        """
        Saves the complete state of the GA to a file atomically.

        Args:
            state: A dictionary containing the GA state
        """
        try:
            state.rng_state = {
                "python_random": random.getstate(),
                "torch_random": torch.get_rng_state(),
                "torch_cuda_random": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }

            # Atomic save: write to temp file first, then rename
            with open(self.temp_filepath, "wb") as f:
                pickle.dump(state, f)
            os.rename(self.temp_filepath, self.filepath)

            fold_info = (
                f" (Fold {state.outer_fold_k + 1})"
                if state.outer_fold_k != -1
                else ""
            )
            logger.info(
                f"Generation {state.generation}/{fold_info} checkpointed successfully."
            )

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(self.temp_filepath):
                os.remove(self.temp_filepath)

    def load_checkpoint(self) -> Optional[GaState]:
        """
        Loads the GA state from a checkpoint file and restores RNG states.
        """
        if not self.is_checkpoint_exists():
            return None

        try:
            with open(self.filepath, "rb") as f:
                state: GaState = pickle.load(f)

            rng_state = state.rng_state
            #random.setstate(rng_state["python_random"])
            #torch.set_rng_state(rng_state["torch_random"])

            #if torch.cuda.is_available() and rng_state.get("torch_cuda_random"):
            #    torch.cuda.set_rng_state_all(rng_state["torch_cuda_random"])

            if not hasattr(state, "outer_fold_k"):
                logger.warning(
                    "Loaded a legacy checkpoint. Disabling nested resampling."
                )
                state.outer_fold_k = -1

            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self) -> None:
        """Deletes the checkpoint file if it exists."""
        if self.is_checkpoint_exists():
            os.remove(self.filepath)


checkpoint_manager = _CheckpointManager()
