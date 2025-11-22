import os
from datetime import datetime

import torch
from torch.nn import Module

from src.utils.file_helper import ensure_dir_exists


class ModelSaver:
    """
    Utility class for saving and loading PyTorch models.
    """

    def __init__(self, filename: str) -> None:
        """
        Args:
            filename: Saved model's filename
        """
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        data_dir = os.environ.get("DATA_DIR", ".")
        self.saved_models_dir = os.path.join(data_dir, task_id, "saved_models")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_time = f"{filename}_{timestamp}.pth"
        self.filepath = os.path.join(self.saved_models_dir, filename_with_time)

    def save(self, model: Module) -> str | None:
        """
        Save the model's state_dict to the file.

        Args:
            model: The PyTorch model to save.
        """
        callback_msg = ensure_dir_exists(self.saved_models_dir)
        torch.save(model.state_dict(), self.filepath)

        return callback_msg

    def load(self, model: Module, map_location: str = "cpu") -> None:
        """
        Load the state_dict from the file into the provided model.

        Args:
            model: The PyTorch model instance to load state into.
            map_location: Device mapping for loading ('cpu', 'cuda', etc.).
        """
        state_dict = torch.load(self.filepath, map_location=map_location)
        model.load_state_dict(state_dict)
