import os
from datetime import datetime

from src.logger.logger import logger


def ensure_dir_exists(filepath: str) -> str | None:
    """Creates the dir if it does not already exist."""
    if filepath and not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
        return f"Created dir '{filepath}'."
    return None


def clear_base_log_file(file_path: str = "./logs/log.log") -> None:
    """
    Clears the contents of a file.
    """
    try:
        with open(file_path, "w"):
            pass
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
    except Exception as e:
        logger.error(f"Error clearing file '{file_path}': {e}")

    return None


def rename_base_log_file(file_path: str = "./logs/log.log") -> str | None:
    """
    Renames the file by appending a timestamp before the extension.
    """
    dir_name, base_name = os.path.split(file_path)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_name = f"{name}_{timestamp}{ext}"
    new_path = os.path.join(dir_name, new_name)

    try:
        os.rename(file_path, new_path)
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
    except Exception as e:
        return logger.error(f"Error renaming file: {e}")

    return None
