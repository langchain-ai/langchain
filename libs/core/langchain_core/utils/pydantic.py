"""Utilities for tests."""

from functools import wraps
from typing import Any, Callable, Dict, Type

from langchain_core.pydantic_v1 import BaseModel, root_validator


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    try:
        import pydantic

        return int(pydantic.__version__.split(".")[0])
    except ImportError:
        return 0


PYDANTIC_MAJOR_VERSION = get_pydantic_major_version()


# How to type hint this?
def pre_init(func: Callable) -> Any:
    """Decorator to run a function before model initialization.

    Args:
        func (Callable): The function to run before model initialization.

    Returns:
        Any: The decorated function.
    """

    @root_validator(pre=True)
    @wraps(func)
    def wrapper(cls: Type[BaseModel], values: Dict[str, Any]) -> Dict[str, Any]:
        """Decorator to run a function before model initialization.

        Args:
            cls (Type[BaseModel]): The model class.
            values (Dict[str, Any]): The values to initialize the model with.

        Returns:
            Dict[str, Any]: The values to initialize the model with.
        """
        # Insert default values
        fields = cls.__fields__
        for name, field_info in fields.items():
            # Check if allow_population_by_field_name is enabled
            # If yes, then set the field name to the alias
            if hasattr(cls, "Config"):
                if hasattr(cls.Config, "allow_population_by_field_name"):
                    if cls.Config.allow_population_by_field_name:
                        if field_info.alias in values:
                            values[name] = values.pop(field_info.alias)

            if name not in values or values[name] is None:
                if not field_info.required:
                    if field_info.default_factory is not None:
                        values[name] = field_info.default_factory()
                    else:
                        values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper
