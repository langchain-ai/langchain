import os
from typing import Any, Dict, Optional


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


from typing import overload


@overload
def get_from_value_or_env(
    value: Optional[str],
    env_key: str,
    *,
    allow_none: bool = False,
    default: Optional[str] = None,
) -> str:
    ...


@overload
def get_from_value_or_env(
    value: Optional[str],
    env_key: str,
    *,
    allow_none: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    ...


def get_from_value_or_env(
    value: Optional[str],
    env_key: str,
    *,
    allow_none: bool = False,
    default: Optional[str] = None,
) -> str:
    """Get the value or the environment corresponding variable.

    Args:
        value: The value to check.
        env_key: The environment variable to check if the value is None.
        default: The default value to use if value is None and the environment does
                 not contain the given key.
        allow_none: Whether to allow None as a value or not. If not, an error will be
                    raised if the value is None.
    """
    if value is not None:
        return value
    elif env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    elif allow_none:
        return None
    else:
        raise ValueError(
            f"Did not find {env_key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass a value for via the init"
        )
