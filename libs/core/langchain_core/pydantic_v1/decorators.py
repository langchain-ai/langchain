from functools import wraps
from typing import Dict, Any

from pydantic import model_validator


def pre_init(func: callable) -> callable:
    """Decorator to run a function before model initialization."""

    @model_validator(mode="before")
    @wraps(func)
    def wrapper(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Decorator to run a function before model initialization."""
        # Insert default values
        for name, field_info in cls.model_fields.items():
            if name not in values:
                if field_info.default_factory is not None:
                    values[name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper
