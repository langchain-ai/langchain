"""Generic utility functions."""
import os
from typing import Any, Dict, List, Optional, Tuple


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    elif env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def register(key: str, _registry: Dict[str, Tuple[Any, List[str]]]) -> Any:
    """Add a class/function to a registry with required keyword arguments.

    ``_registry`` is a dictionary mapping from a key to a tuple of the class/function
    and a list of required keyword arguments.
    """

    def _register_cls(cls: Any, required_kwargs: List) -> Any:
        if key in _registry:
            raise ValueError(f"{_registry[key][0]} already registered as {key}")
        _registry[key] = (cls, required_kwargs)
        return cls

    return _register_cls
