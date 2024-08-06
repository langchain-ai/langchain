"""Utilities for tests."""

from functools import wraps
from typing import Any, Callable, Dict, Type

from langchain_core.pydantic_v1 import BaseModel, root_validator
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema


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
            if hasattr(cls, "model_config"):
                if cls.model_config.get("populate_by_name"):
                    if field_info.alias in values:
                        values[name] = values.pop(field_info.alias)

            if name not in values or values[name] is None:
                if not field_info.is_required():
                    if field_info.default_factory is not None:
                        values[name] = field_info.default_factory()
                    else:
                        values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper


class _IgnoreUnserializable(GenerateJsonSchema):
    """A JSON schema generator that ignores unknown types.

    https://docs.pydantic.dev/latest/concepts/json_schema/#customizing-the-json-schema-generation-process
    """

    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        return {}


def v1_repr(obj: BaseModel) -> str:
    """Return the schema of the object as a string.

    Get a repr for the pydantic object which is consistent with pydantic.v1.
    """
    if not isinstance(obj, BaseModel):
        raise TypeError(f"Expected a pydantic BaseModel, got {type(obj)}")
    repr_ = []
    for name, field in obj.__fields__.items():
        value = getattr(obj, name)

        if isinstance(value, BaseModel):
            repr_.append(f"{name}={v1_repr(value)}")
        else:
            if not field.is_required():
                if not value:
                    continue
                if field.default == value:
                    continue

            repr_.append(f"{name}={repr(value)}")

    args = ", ".join(repr_)
    return f"{obj.__class__.__name__}({args})"
