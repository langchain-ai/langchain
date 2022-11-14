"""Common utility functions for working with LLM APIs."""
import os
import re
from typing import Any, Dict, List


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text)[0]


def get_from_dict_or_env(data: Dict[str, Any], key: str, env_key: str) -> Any:
    """Get a value from a dictionary or an environment variable."""
    if key in data:
        return data[key]
    return os.environ.get(env_key, None)
