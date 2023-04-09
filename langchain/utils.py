"""Generic utility functions."""
import os
from typing import Any, Dict, Optional
from dotenv import dotenv_values

_environ = {
    **dotenv_values(),  # load .env file
    **os.environ,       # override loaded values with environment variables
}


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    elif env_key in _environ and _environ[env_key]:
        return _environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, add `{env_key}=...` to .env file or pass"
            f"  `{key}` as a named parameter."
        )
