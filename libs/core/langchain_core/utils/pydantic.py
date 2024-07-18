"""Utilities for tests."""

from __future__ import annotations

import inspect
import textwrap
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

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


def _create_subset_model_v1(
    name: str,
    model: Type[BaseModel],
    field_names: list,
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    from langchain_core.pydantic_v1 import create_model

    fields = {}

    for field_name in field_names:
        field = model.__fields__[field_name]
        t = (
            # this isn't perfect but should work for most functions
            field.outer_type_
            if field.required and not field.allow_none
            else Optional[field.outer_type_]
        )
        if descriptions and field_name in descriptions:
            field.field_info.description = descriptions[field_name]
        fields[field_name] = (t, field.field_info)

    rtn = create_model(name, **fields)  # type: ignore
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


def _create_subset_model_v2(
    name: str,
    model: Type[BaseModel],
    field_names: List[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[BaseModel]:
    """Create a pydantic model with a subset of the model fields."""
    from pydantic import create_model  # pydantic: ignore
    from pydantic.fields import FieldInfo  # pydantic: ignore

    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]  # type: ignore
        description = descriptions_.get(field_name, field.description)
        fields[field_name] = (
            field.annotation,
            FieldInfo(description=description, default=field.default),
        )
    rtn = create_model(name, **fields)  # type: ignore

    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


# Private functionality to create a subset model that's compatible across
# different versions of pydantic.
# Handles pydantic versions 1.x and 2.x. including v1 of pydantic in 2.x.
# However, can't find a way to type hint this.
def _create_subset_model(
    name: str,
    model: Type[BaseModel],
    field_names: List[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[BaseModel]:
    """Create subset model using the same pydantic version as the input model."""
    if PYDANTIC_MAJOR_VERSION == 1:
        return _create_subset_model_v1(
            name,
            model,
            field_names,
            descriptions=descriptions,
            fn_description=fn_description,
        )
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic.v1 import BaseModel as BaseModelV1  # pydantic: ignore

        if issubclass(model, BaseModelV1):
            return _create_subset_model_v1(
                name,
                model,
                field_names,
                descriptions=descriptions,
                fn_description=fn_description,
            )
        else:
            return _create_subset_model_v2(
                name,
                model,
                field_names,
                descriptions=descriptions,
                fn_description=fn_description,
            )
    else:
        raise NotImplementedError(
            f"Unsupported pydantic version: {PYDANTIC_MAJOR_VERSION}"
        )
