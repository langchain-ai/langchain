"""Generic utility functions."""
import os
from typing import Any, Dict


def get_from_dict_or_env(data: Dict[str, Any], key: str, env_key: str) -> Any:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    return os.environ.get(env_key, None)
