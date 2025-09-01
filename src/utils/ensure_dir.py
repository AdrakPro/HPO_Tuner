import os

from src.logger.experiment_logger import logger


def ensure_dir_exists(filepath: str):
    """Creates the dir if it does not already exist."""
    if filepath and not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
        logger.info(f"Created dir '{filepath}'.")
