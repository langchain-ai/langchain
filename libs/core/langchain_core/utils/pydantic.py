"""Utilities for pydantic."""

from __future__ import annotations

import textwrap
import warnings
from contextlib import nullcontext
from functools import lru_cache, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    cast,
)

from packaging import version
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic import (
    create_model as _create_model_base,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
    JsonSchemaValue,
)
from pydantic.version import VERSION as PYDANTIC_VERSION_STRING
from typing_extensions import override

if TYPE_CHECKING:
    from pydantic_core import core_schema

PYDANTIC_VERSION = version.parse(PYDANTIC_VERSION_STRING)
PYDANTIC_MINOR_VERSION = PYDANTIC_VERSION.minor


# How to type hint this?
def pre_init(func: Callable) -> Any:
    """Decorator to run a function before model initialization.

    Args:
        func (Callable): The function to run before model initialization.

    Returns:
        Any: The decorated function.
    """

    @model_validator(mode="before")
    @wraps(func)
    def wrapper(cls: type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
        """Decorator to run a function before model initialization.

        Args:
            cls (Type[BaseModel]): The model class.
            values (dict[str, Any]): The values to initialize the model with.

        Returns:
            dict[str, Any]: The values to initialize the model with.
        """
        # Insert default values
        fields = cls.model_fields
        for name, field_info in fields.items():
            if cls.model_config.get("populate_by_name") and field_info.alias in values:
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


def _create_subset_model_v2(
    name: str,
    model: type[BaseModel],
    field_names: list[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> type[BaseModel]:
    """Create a pydantic model with a subset of the model fields."""
    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]
        description = descriptions_.get(field_name, field.description)
        field_info = FieldInfo(description=description, default=field.default)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)

    rtn = _create_model_base(
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


def _create_subset_model(
    name: str,
    model: type[BaseModel],
    field_names: list[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> type[BaseModel]:
    """Create subset model using the same pydantic version as the input model."""
    return _create_subset_model_v2(
        name,
        model,
        field_names,
        descriptions=descriptions,
        fn_description=fn_description,
    )


def get_fields(
    model: type[BaseModel],
) -> dict[str, FieldInfo]:
    """Get the field names of a Pydantic model."""
    try:
        return model.model_fields
    except AttributeError as exc:
        msg = f"Expected a Pydantic model. Got {type(model)}"
        raise TypeError(msg) from exc


_SchemaConfig = ConfigDict(
    arbitrary_types_allowed=True, frozen=True, protected_namespaces=()
)

NO_DEFAULT = object()


def _create_root_model(
    name: str,
    type_: Any,
    module_name: Optional[str] = None,
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
        custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast("type[BaseModel]", custom_root_type)


@lru_cache(maxsize=256)
def _create_root_model_cached(
    model_name: str,
    type_: Any,
    *,
    module_name: Optional[str] = None,
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
    module_name: Optional[str] = None,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    """Create a pydantic model with the given field definitions.

    Please use create_model_v2 instead of this function.

    Args:
        model_name: The name of the model.
        module_name: The name of the module where the model is defined.
            This is used by Pydantic to resolve any forward references.
        **field_definitions: The field definitions for the model.

    Returns:
        Type[BaseModel]: The created model.
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
            if isinstance(value, FieldInfo):
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
    module_name: Optional[str] = None,
    field_definitions: Optional[dict[str, Any]] = None,
    root: Optional[Any] = None,
) -> type[BaseModel]:
    """Create a pydantic model with the given field definitions.

    Attention:
        Please do not use outside of langchain packages. This API
        is subject to change at any time.

    Args:
        model_name: The name of the model.
        module_name: The name of the module where the model is defined.
            This is used by Pydantic to resolve any forward references.
        field_definitions: The field definitions for the model.
        root: Type for a root model (RootModel)

    Returns:
        type[BaseModel]: The created model.
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
