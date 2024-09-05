"""Utilities for tests."""

from __future__ import annotations

import inspect
import textwrap
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, overload

import pydantic
from pydantic import BaseModel, root_validator
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


if PYDANTIC_MAJOR_VERSION == 1:
    from pydantic.fields import FieldInfo as FieldInfoV1

    PydanticBaseModel = pydantic.BaseModel
    TypeBaseModel = Type[BaseModel]
elif PYDANTIC_MAJOR_VERSION == 2:
    from pydantic.v1.fields import FieldInfo as FieldInfoV1  # type: ignore[assignment]

    # Union type needs to be last assignment to PydanticBaseModel to make mypy happy.
    PydanticBaseModel = Union[BaseModel, pydantic.BaseModel]  # type: ignore
    TypeBaseModel = Union[Type[BaseModel], Type[pydantic.BaseModel]]  # type: ignore
else:
    raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")


TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


def is_pydantic_v1_subclass(cls: Type) -> bool:
    """Check if the installed Pydantic version is 1.x-like."""
    if PYDANTIC_MAJOR_VERSION == 1:
        return True
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic.v1 import BaseModel as BaseModelV1

        if issubclass(cls, BaseModelV1):
            return True
    return False


def is_pydantic_v2_subclass(cls: Type) -> bool:
    """Check if the installed Pydantic version is 1.x-like."""
    from pydantic import BaseModel

    return PYDANTIC_MAJOR_VERSION == 2 and issubclass(cls, BaseModel)


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
        fields = cls.model_fields
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
    if not is_basemodel_instance(obj):
        raise TypeError(f"Expected a pydantic BaseModel, got {type(obj)}")
    repr_ = []
    for name, field in get_fields(obj).items():
        value = getattr(obj, name)

        if isinstance(value, BaseModel):
            repr_.append(f"{name}={v1_repr(value)}")
        else:
            if field.exclude:
                continue
            if not field.is_required():
                if not value:
                    continue
                if field.default == value:
                    continue

            repr_.append(f"{name}={repr(value)}")

    args = ", ".join(repr_)
    return f"{obj.__class__.__name__}({args})"


def _create_subset_model_v1(
    name: str,
    model: Type[BaseModel],
    field_names: list,
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    if PYDANTIC_MAJOR_VERSION == 1:
        from pydantic import create_model
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic.v1 import create_model  # type: ignore
    else:
        raise NotImplementedError(
            f"Unsupported pydantic version: {PYDANTIC_MAJOR_VERSION}"
        )

    fields = {}

    for field_name in field_names:
        # Using pydantic v1 so can access __fields__ as a dict.
        field = model.__fields__[field_name]  # type: ignore
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
    model: Type[pydantic.BaseModel],
    field_names: List[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[pydantic.BaseModel]:
    """Create a pydantic model with a subset of the model fields."""
    from pydantic import create_model
    from pydantic.fields import FieldInfo

    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]  # type: ignore
        description = descriptions_.get(field_name, field.description)
        field_info = FieldInfo(description=description, default=field.default)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)
    rtn = create_model(name, **fields)  # type: ignore

    # TODO(0.3): Determine if there is a more "pydantic" way to preserve annotations.
    # This is done to preserve __annotations__ when working with pydantic 2.x
    # and using the Annotated type with TypedDict.
    # Comment out the following line, to trigger the relevant test case.
    selected_annotations = [
        (name, annotation)
        for name, annotation in model.__annotations__.items()
        if name in field_names
    ]

    rtn.__annotations__ = dict(selected_annotations)
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


# Private functionality to create a subset model that's compatible across
# different versions of pydantic.
# Handles pydantic versions 1.x and 2.x. including v1 of pydantic in 2.x.
# However, can't find a way to type hint this.
def _create_subset_model(
    name: str,
    model: TypeBaseModel,
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
        from pydantic.v1 import BaseModel as BaseModelV1

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


if PYDANTIC_MAJOR_VERSION == 2:
    from pydantic import BaseModel as BaseModelV2
    from pydantic.fields import FieldInfo as FieldInfoV2
    from pydantic.v1 import BaseModel as BaseModelV1

    @overload
    def get_fields(model: Type[BaseModelV2]) -> Dict[str, FieldInfoV2]: ...

    @overload
    def get_fields(model: BaseModelV2) -> Dict[str, FieldInfoV2]: ...

    @overload
    def get_fields(model: Type[BaseModelV1]) -> Dict[str, FieldInfoV1]: ...

    @overload
    def get_fields(model: BaseModelV1) -> Dict[str, FieldInfoV1]: ...

    def get_fields(
        model: Union[
            BaseModelV2,
            BaseModelV1,
            Type[BaseModelV2],
            Type[BaseModelV1],
        ],
    ) -> Union[Dict[str, FieldInfoV2], Dict[str, FieldInfoV1]]:
        """Get the field names of a Pydantic model."""
        if hasattr(model, "model_fields"):
            return model.model_fields  # type: ignore

        elif hasattr(model, "__fields__"):
            return model.__fields__  # type: ignore
        else:
            raise TypeError(f"Expected a Pydantic model. Got {type(model)}")
elif PYDANTIC_MAJOR_VERSION == 1:
    from pydantic import BaseModel as BaseModelV1_

    def get_fields(  # type: ignore[no-redef]
        model: Union[Type[BaseModelV1_], BaseModelV1_],
    ) -> Dict[str, FieldInfoV1]:
        """Get the field names of a Pydantic model."""
        return model.__fields__  # type: ignore
else:
    raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")
