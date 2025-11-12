"""Utilities for pydantic."""

from __future__ import annotations

import inspect
import textwrap
import warnings
from contextlib import nullcontext
from functools import lru_cache, wraps
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import pydantic
from packaging import version

# root_validator is deprecated but we need it for backward compatibility of @pre_init
from pydantic import (  # type: ignore[deprecated]
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecationWarning,
    RootModel,
    root_validator,
)
from pydantic import (
    create_model as _create_model_base,
)
from pydantic.fields import FieldInfo as FieldInfoV2
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
    JsonSchemaValue,
)
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import create_model as create_model_v1
from typing_extensions import deprecated, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.v1.fields import ModelField
    from pydantic_core import core_schema

PYDANTIC_VERSION = version.parse(pydantic.__version__)


@deprecated("Use PYDANTIC_VERSION.major instead.")
def get_pydantic_major_version() -> int:
    """DEPRECATED - Get the major version of Pydantic.

    Use PYDANTIC_VERSION.major instead.

    Returns:
        The major version of Pydantic.
    """
    return PYDANTIC_VERSION.major


PYDANTIC_MAJOR_VERSION = PYDANTIC_VERSION.major
PYDANTIC_MINOR_VERSION = PYDANTIC_VERSION.minor

IS_PYDANTIC_V1 = False
IS_PYDANTIC_V2 = True

PydanticBaseModel = BaseModel
TypeBaseModel = type[BaseModel]

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


def is_pydantic_v1_subclass(cls: type) -> bool:
    """Check if the given class is Pydantic v1-like.

    Returns:
        `True` if the given class is a subclass of Pydantic `BaseModel` 1.x.
    """
    return issubclass(cls, BaseModelV1)


def is_pydantic_v2_subclass(cls: type) -> bool:
    """Check if the given class is Pydantic v2-like.

    Returns:
        `True` if the given class is a subclass of Pydantic BaseModel 2.x.
    """
    return issubclass(cls, BaseModel)


def is_basemodel_subclass(cls: type) -> bool:
    """Check if the given class is a subclass of Pydantic BaseModel.

    Check if the given class is a subclass of any of the following:

    * pydantic.BaseModel in Pydantic 2.x
    * pydantic.v1.BaseModel in Pydantic 2.x

    Returns:
        `True` if the given class is a subclass of Pydantic `BaseModel`.
    """
    # Before we can use issubclass on the cls we need to check if it is a class
    if not inspect.isclass(cls) or isinstance(cls, GenericAlias):
        return False

    return issubclass(cls, (BaseModel, BaseModelV1))


def is_basemodel_instance(obj: Any) -> bool:
    """Check if the given class is an instance of Pydantic BaseModel.

    Check if the given class is an instance of any of the following:

    * pydantic.BaseModel in Pydantic 2.x
    * pydantic.v1.BaseModel in Pydantic 2.x

    Returns:
        `True` if the given class is an instance of Pydantic `BaseModel`.
    """
    return isinstance(obj, (BaseModel, BaseModelV1))


# How to type hint this?
def pre_init(func: Callable) -> Any:
    """Decorator to run a function before model initialization.

    Args:
        func: The function to run before model initialization.

    Returns:
        The decorated function.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PydanticDeprecationWarning)

        # Ideally we would use @model_validator(mode="before") but this would change the
        # order of the validators. See https://github.com/pydantic/pydantic/discussions/7434.
        # So we keep root_validator for backward compatibility.
        @root_validator(pre=True)  # type: ignore[deprecated]
        @wraps(func)
        def wrapper(cls: type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
            """Decorator to run a function before model initialization.

            Args:
                cls: The model class.
                values: The values to initialize the model with.

            Returns:
                The values to initialize the model with.
            """
            # Insert default values
            fields = cls.model_fields
            for name, field_info in fields.items():
                # Check if allow_population_by_field_name is enabled
                # If yes, then set the field name to the alias
                if (
                    hasattr(cls, "Config")
                    and hasattr(cls.Config, "allow_population_by_field_name")
                    and cls.Config.allow_population_by_field_name
                    and field_info.alias in values
                ):
                    values[name] = values.pop(field_info.alias)
                if (
                    hasattr(cls, "model_config")
                    and cls.model_config.get("populate_by_name")
                    and field_info.alias in values
                ):
                    values[name] = values.pop(field_info.alias)

                if (
                    name not in values or values[name] is None
                ) and not field_info.is_required():
                    if field_info.default_factory is not None:
                        values[name] = field_info.default_factory()  # type: ignore[call-arg]
                    else:
                        values[name] = field_info.default

            # Call the decorated function
            return func(cls, values)

    return wrapper


class _IgnoreUnserializable(GenerateJsonSchema):
    """A JSON schema generator that ignores unknown types.

    https://docs.pydantic.dev/latest/concepts/json_schema/#customizing-the-json-schema-generation-process
    """

    @override
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        return {}


def _create_subset_model_v1(
    name: str,
    model: type[BaseModelV1],
    field_names: list,
    *,
    descriptions: dict | None = None,
    fn_description: str | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model with only a subset of model's fields."""
    fields = {}

    for field_name in field_names:
        # Using pydantic v1 so can access __fields__ as a dict.
        field = model.__fields__[field_name]
        t = (
            # this isn't perfect but should work for most functions
            field.outer_type_
            if field.required and not field.allow_none
            else field.outer_type_ | None
        )
        if descriptions and field_name in descriptions:
            field.field_info.description = descriptions[field_name]
        fields[field_name] = (t, field.field_info)

    rtn = create_model_v1(name, **fields)  # type: ignore[call-overload]
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


def _create_subset_model_v2(
    name: str,
    model: type[BaseModel],
    field_names: list[str],
    *,
    descriptions: dict | None = None,
    fn_description: str | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model with a subset of the model fields."""
    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]
        description = descriptions_.get(field_name, field.description)
        field_info = FieldInfoV2(description=description, default=field.default)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)

    rtn = _create_model_base(  # type: ignore[call-overload]
        name, **fields, __config__=ConfigDict(arbitrary_types_allowed=True)
    )

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
# Handles pydantic versions 2.x. including v1 of pydantic in 2.x.
# However, can't find a way to type hint this.
def _create_subset_model(
    name: str,
    model: TypeBaseModel,
    field_names: list[str],
    *,
    descriptions: dict | None = None,
    fn_description: str | None = None,
) -> type[BaseModel]:
    """Create subset model using the same pydantic version as the input model.

    Returns:
        The created subset model.
    """
    if issubclass(model, BaseModelV1):
        return _create_subset_model_v1(
            name,
            model,
            field_names,
            descriptions=descriptions,
            fn_description=fn_description,
        )
    return _create_subset_model_v2(
        name,
        model,
        field_names,
        descriptions=descriptions,
        fn_description=fn_description,
    )


@overload
def get_fields(model: type[BaseModel]) -> dict[str, FieldInfoV2]: ...


@overload
def get_fields(model: BaseModel) -> dict[str, FieldInfoV2]: ...


@overload
def get_fields(model: type[BaseModelV1]) -> dict[str, ModelField]: ...


@overload
def get_fields(model: BaseModelV1) -> dict[str, ModelField]: ...


def get_fields(
    model: type[BaseModel | BaseModelV1] | BaseModel | BaseModelV1,
) -> dict[str, FieldInfoV2] | dict[str, ModelField]:
    """Return the field names of a Pydantic model.

    Args:
        model: The Pydantic model or instance.

    Raises:
        TypeError: If the model is not a Pydantic model.
    """
    if not isinstance(model, type):
        model = type(model)
    if issubclass(model, BaseModel):
        return model.model_fields
    if issubclass(model, BaseModelV1):
        return model.__fields__
    msg = f"Expected a Pydantic model. Got {model}"
    raise TypeError(msg)


_SchemaConfig = ConfigDict(
    arbitrary_types_allowed=True, frozen=True, protected_namespaces=()
)

NO_DEFAULT = object()


def _create_root_model(
    name: str,
    type_: Any,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    """Create a base class."""

    def schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
    ) -> dict[str, Any]:
        # Complains about schema not being defined in superclass
        schema_ = super(cls, cls).schema(  # type: ignore[misc]
            by_alias=by_alias, ref_template=ref_template
        )
        schema_["title"] = name
        return schema_

    def model_json_schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        # Complains about model_json_schema not being defined in superclass
        schema_ = super(cls, cls).model_json_schema(  # type: ignore[misc]
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
        )
        schema_["title"] = name
        return schema_

    base_class_attributes = {
        "__annotations__": {"root": type_},
        "model_config": ConfigDict(arbitrary_types_allowed=True),
        "schema": classmethod(schema),
        "model_json_schema": classmethod(model_json_schema),
        "__module__": module_name or "langchain_core.runnables.utils",
    }

    if default_ is not NO_DEFAULT:
        base_class_attributes["root"] = default_
    with warnings.catch_warnings():
        try:
            if (
                isinstance(type_, type)
                and not isinstance(type_, GenericAlias)
                and issubclass(type_, BaseModelV1)
            ):
                warnings.filterwarnings(
                    action="ignore", category=PydanticDeprecationWarning
                )
        except TypeError:
            pass
        custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast("type[BaseModel]", custom_root_type)


@lru_cache(maxsize=256)
def _create_root_model_cached(
    model_name: str,
    type_: Any,
    *,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    return _create_root_model(
        model_name, type_, default_=default_, module_name=module_name
    )


@lru_cache(maxsize=256)
def _create_model_cached(
    model_name: str,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    return _create_model_base(
        model_name,
        __config__=_SchemaConfig,
        **_remap_field_definitions(field_definitions),
    )


def create_model(
    model_name: str,
    module_name: str | None = None,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    """Create a Pydantic model with the given field definitions.

    Please use `create_model_v2` instead of this function.

    Args:
        model_name: The name of the model.
        module_name: The name of the module where the model is defined.
            This is used by Pydantic to resolve any forward references.
        **field_definitions: The field definitions for the model.

    Returns:
        The created model.
    """
    kwargs = {}
    if "__root__" in field_definitions:
        kwargs["root"] = field_definitions.pop("__root__")

    return create_model_v2(
        model_name,
        module_name=module_name,
        field_definitions=field_definitions,
        **kwargs,
    )


# Reserved names should capture all the `public` names / methods that are
# used by BaseModel internally. This will keep the reserved names up-to-date.
# For reference, the reserved names are:
# "construct", "copy", "dict", "from_orm", "json", "parse_file", "parse_obj",
# "parse_raw", "schema", "schema_json", "update_forward_refs", "validate",
# "model_computed_fields", "model_config", "model_construct", "model_copy",
# "model_dump", "model_dump_json", "model_extra", "model_fields",
# "model_fields_set", "model_json_schema", "model_parametrized_name",
# "model_post_init", "model_rebuild", "model_validate", "model_validate_json",
# "model_validate_strings"
_RESERVED_NAMES = {key for key in dir(BaseModel) if not key.startswith("_")}


def _remap_field_definitions(field_definitions: dict[str, Any]) -> dict[str, Any]:
    """This remaps fields to avoid colliding with internal pydantic fields."""
    remapped = {}
    for key, value in field_definitions.items():
        if key.startswith("_") or key in _RESERVED_NAMES:
            # Let's add a prefix to avoid colliding with internal pydantic fields
            if isinstance(value, FieldInfoV2):
                msg = (
                    f"Remapping for fields starting with '_' or fields with a name "
                    f"matching a reserved name {_RESERVED_NAMES} is not supported if "
                    f" the field is a pydantic Field instance. Got {key}."
                )
                raise NotImplementedError(msg)
            type_, default_ = value
            remapped[f"private_{key}"] = (
                type_,
                Field(
                    default=default_,
                    alias=key,
                    serialization_alias=key,
                    title=key.lstrip("_").replace("_", " ").title(),
                ),
            )
        else:
            remapped[key] = value
    return remapped


def create_model_v2(
    model_name: str,
    *,
    module_name: str | None = None,
    field_definitions: dict[str, Any] | None = None,
    root: Any | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model with the given field definitions.

    Attention:
        Please do not use outside of langchain packages. This API
        is subject to change at any time.

    Args:
        model_name: The name of the model.
        module_name: The name of the module where the model is defined.
            This is used by Pydantic to resolve any forward references.
        field_definitions: The field definitions for the model.
        root: Type for a root model (`RootModel`)

    Returns:
        The created model.
    """
    field_definitions = field_definitions or {}

    if root:
        if field_definitions:
            msg = (
                "When specifying __root__ no other "
                f"fields should be provided. Got {field_definitions}"
            )
            raise NotImplementedError(msg)

        if isinstance(root, tuple):
            kwargs = {"type_": root[0], "default_": root[1]}
        else:
            kwargs = {"type_": root}

        try:
            named_root_model = _create_root_model_cached(
                model_name, module_name=module_name, **kwargs
            )
        except TypeError:
            # something in the arguments into _create_root_model_cached is not hashable
            named_root_model = _create_root_model(
                model_name,
                module_name=module_name,
                **kwargs,
            )
        return named_root_model

    # No root, just field definitions
    names = set(field_definitions.keys())

    capture_warnings = False

    for name in names:
        # Also if any non-reserved name is used (e.g., model_id or model_name)
        if name.startswith("model"):
            capture_warnings = True

    with warnings.catch_warnings() if capture_warnings else nullcontext():
        if capture_warnings:
            warnings.filterwarnings(action="ignore")
        try:
            return _create_model_cached(model_name, **field_definitions)
        except TypeError:
            # something in field definitions is not hashable
            return _create_model_base(
                model_name,
                __config__=_SchemaConfig,
                **_remap_field_definitions(field_definitions),
            )
