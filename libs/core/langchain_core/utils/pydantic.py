"""Utilities for tests."""
from functools import wraps
from typing import Dict, Any

from langchain_core.pydantic_v1 import root_validator


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    try:
        import pydantic

        return int(pydantic.__version__.split(".")[0])
    except ImportError:
        return 0


PYDANTIC_MAJOR_VERSION = get_pydantic_major_version()


def pre_init(func: callable) -> callable:
    """Decorator to run a function before model initialization."""

    @root_validator(pre=True)
    @wraps(func)
    def wrapper(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Decorator to run a function before model initialization."""
        # Insert default values
        fields = cls.__fields__
        for name, field_info in fields.items():
            if name not in values or values[name] is None:
                if field_info.default_factory is not None:
                    values[name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper
