"""Utilities for tests."""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Type, Union

from langchain_core.pydantic_v1 import BaseModel, root_validator


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    try:
        import pydantic

        return int(pydantic.__version__.split(".")[0])
    except ImportError:
        return 0


PYDANTIC_MAJOR_VERSION = get_pydantic_major_version()


def is_basemodel_subclass(cls: Type) -> bool:
    """Check if the given class is a subclass of Pydantic BaseModel.

    Check if the given class is a subclass of any of the following:

    * pydantic.BaseModel in Pydantic 1.x
    * pydantic.BaseModel in Pydantic 2.x
    * pydantic.v1.BaseModel in Pydantic 2.x
    """
    # Before we can use issubclass on the cls we need to check if it is a class
    if not inspect.isclass(cls):
        return False

    if PYDANTIC_MAJOR_VERSION == 1:
        from pydantic import BaseModel as BaseModelV1Proper

        if issubclass(cls, BaseModelV1Proper):
            return True
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic import BaseModel as BaseModelV2
        from pydantic.v1 import BaseModel as BaseModelV1

        if issubclass(cls, BaseModelV2):
            return True

        if issubclass(cls, BaseModelV1):
            return True
    else:
        raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")
    return False

def is_basemodel_instance(obj: Any) -> bool:
    """Check if the given class is an instance of Pydantic BaseModel.

    Check if the given class is an instance of any of the following:

    * pydantic.BaseModel in Pydantic 1.x
    * pydantic.BaseModel in Pydantic 2.x
    * pydantic.v1.BaseModel in Pydantic 2.x
    """
    if PYDANTIC_MAJOR_VERSION == 1:
        from pydantic import BaseModel as BaseModelV1Proper

        if isinstance(obj, BaseModelV1Proper):
            return True
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic import BaseModel as BaseModelV2
        from pydantic.v1 import BaseModel as BaseModelV1

        if isinstance(obj, BaseModelV2):
            return True

        if isinstance(obj, BaseModelV1):
            return True
    else:
        raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")
    return False


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
