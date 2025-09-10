import os
import pickle
import random
from typing import Any, Optional, List, Dict
from dataclasses import dataclass

import torch

from src.logger.logger import logger


@dataclass
class GaState:
    generation: int
    population: List[Dict]
    fitness_scores: List[float]
    phase: str
    config: Dict
    session_log_filename: str
    phase_completed: bool
    rng_state: Optional[Dict[str, Any]] = None


class _CheckpointManager:
    """
    Manages saving and loading of the application state for resuming interrupted runs.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        filename: str = "ga_checkpoint.pkl",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.filepath = os.path.join(self.checkpoint_dir, filename)
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
            logger.info(
                f"Generation {state.generation} saved successfully to {self.filepath}"
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
            random.setstate(rng_state["python_random"])
            torch.set_rng_state(rng_state["torch_random"])

            if torch.cuda.is_available() and rng_state.get("torch_cuda_random"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda_random"])

            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self) -> None:
        """Deletes the checkpoint file if it exists."""
        if self.is_checkpoint_exists():
            os.remove(self.filepath)


checkpoint_manager = _CheckpointManager()
