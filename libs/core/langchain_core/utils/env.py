"""Utilities for environment variables."""

from __future__ import annotations

import os
from typing import Any, Optional, Union


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return env_var in os.environ and os.environ[env_var] not in {
        "",
        "0",
        "false",
        "False",
    }


def get_from_dict_or_env(
    data: dict[str, Any],
    key: Union[str, list[str]],
    env_key: str,
    default: Optional[str] = None,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary. This can be a list of keys to try
            in order.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.

    Returns:
        The dict value or the environment variable value.
    """
    if isinstance(key, (list, tuple)):
        for k in key:
            if value := data.get(k):
                return value

    if isinstance(key, str) and key in data and data[key]:
        return data[key]

    key_for_err = key[0] if isinstance(key, (list, tuple)) else key

    return get_from_env(key_for_err, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        key: The key to look up in the dictionary.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.

    Returns:
        str: The value of the key.

    Raises:
        ValueError: If the key is not in the dictionary and no default value is
            provided or if the environment variable is not set.
    """
    if env_value := os.getenv(env_key):
        return env_value
    if default is not None:
        return default
    msg = (
        f"Did not find {key}, please add an environment variable"
        f" `{env_key}` which contains it, or pass"
        f" `{key}` as a named parameter."
    )
    raise ValueError(msg)
