import os
from datetime import datetime

import torch
from torch.nn import Module

from src.utils.ensure_dir import ensure_dir_exists


# TODO: problem, should we save every training or just that with the best params given by GA (or just keep it, rest delete)?
class ModelSaver:
    """
    Utility class for saving and loading PyTorch models.
    """

    def __init__(self, filename: str) -> None:
        """
        Args:
            filename: Saved model's filename
        """
        self.dir = f"./saved_models"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_time = f"{filename}_{timestamp}.pth"
        self.filepath = os.path.join(self.dir, filename_with_time)

    def save(self, model: Module) -> None:
        """
        Save the model's state_dict to the file.

        Args:
            model: The PyTorch model to save.
        """
        ensure_dir_exists(self.dir)
        torch.save(model.state_dict(), self.filepath)

    def load(self, model: Module, map_location: str = "cpu") -> None:
        """
        Load the state_dict from the file into the provided model.

        Args:
            model: The PyTorch model instance to load state into.
            map_location: Device mapping for loading ('cpu', 'cuda', etc.).
        """
        state_dict = torch.load(self.filepath, map_location=map_location)
        model.load_state_dict(state_dict)
