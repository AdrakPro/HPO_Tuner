from typing import Dict, Any, List


def get_nested_config(cfg: Dict, path: List[str], default: Any = None) -> Any:
    """Safely retrieve a nested value from a dictionary."""
    for key in path:
        if not isinstance(cfg, dict) or key not in cfg:
            return default
        cfg = cfg[key]
    return cfg
