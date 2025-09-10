import os


def ensure_dir_exists(filepath: str) -> str | None:
    """Creates the dir if it does not already exist."""
    if filepath and not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
        return f"Created dir '{filepath}'."
    return None
