import typing
from typing import Any, List, Optional, Sequence, Type, Union, get_args, get_origin

from langchain.load.serializable import Serializable
from langchain.schema.runnable import RunnableConfig

try:
    from pydantic.v1 import BaseModel, Field, create_model
except ImportError:
    from pydantic import BaseModel, Field, create_model, validator

from typing_extensions import TypedDict

InputValidator = Union[Type[BaseModel], type]
# The following langchain objects are considered to be safe to load.

# PUBLIC API


def create_runnable_config_model(
    ns: str, config_keys: Sequence[str]
) -> type(TypedDict):
    """Create a projection of the runnable config type.

    Args:
        ns: The namespace of the runnable config type.
        config_keys: The keys to include in the projection.
    """
    subset_dict = {}
    for key in config_keys:
        if key in RunnableConfig.__annotations__:
            subset_dict[key] = RunnableConfig.__annotations__[key]
        else:
            raise AssertionError(f"Key {key} not in RunnableConfig.")

    return TypedDict(f"{ns}RunnableConfig", subset_dict, total=False)


def create_invoke_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    invoke_request_type = create_model(
        f"{namespace}InvokeRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    invoke_request_type.update_forward_refs()
    return invoke_request_type


def create_stream_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_request_model = create_model(
        f"{namespace}StreamRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    stream_request_model.update_forward_refs()
    return stream_request_model


def create_batch_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch request."""
    batch_request_type = create_model(
        f"{namespace}BatchRequest",
        inputs=(List[input_type], ...),
        config=(Union[config, List[config]], Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    batch_request_type.update_forward_refs()
    return batch_request_type


def create_stream_log_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_log_request = create_model(
        f"{namespace}StreamLogRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        include_names=(Optional[Sequence[str]], None),
        include_types=(Optional[Sequence[str]], None),
        include_tags=(Optional[Sequence[str]], None),
        exclude_names=(Optional[Sequence[str]], None),
        exclude_types=(Optional[Sequence[str]], None),
        exclude_tags=(Optional[Sequence[str]], None),
        kwargs=(dict, Field(default_factory=dict)),
    )
    stream_log_request.update_forward_refs()
    return stream_log_request


_TYPE_REGISTRY = {}
_SEEN_NAMES = set()


def _create_lc_object_validator(expected_id: Sequence[str]) -> Type[BaseModel]:
    """Create a validator for lc objects.

    An LCObject is used to validate LangChain objects in dict representation.

    The model is associated with a validator that checks that the id of the LCObject
    matches the expected id. This is used to ensure that the LCObject is of the
    correct type.

    For OpenAPI docs to work, each unique LCObject must have a unique name.
    The models are added to the registry to avoid creating duplicate models.

    Args:
        model_id: The expected id of the LCObject.

    Returns:
        A pydantic model that can be used to validate LCObjects.
    """
    expected_id = tuple(expected_id)
    model_id = tuple(["pydantic"]) + expected_id
    if model_id in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[model_id]

    model_name = model_id[-1]

    if model_name in _SEEN_NAMES:
        # Use fully qualified name
        _name = ".".join(model_id)
    else:
        _name = model_name
        if _name in _SEEN_NAMES:
            raise AssertionError(f"Duplicate model name: {_name}")

    _SEEN_NAMES.add(model_name)

    class LCObject(BaseModel):
        id: List[str]
        lc: Any
        type: str
        kwargs: Any

        @validator("id", allow_reuse=True)
        def validate_id_namespace(cls, id: Sequence[str]) -> None:
            """Validate that the LCObject is one of the allowed types."""
            if tuple(id) != expected_id:
                raise ValueError(f"LCObject id {id} is not allowed: {model_id}")
            return id

    # Update the name of the model to make it unique.
    model = create_model(_name, __base__=LCObject)

    _TYPE_REGISTRY[model_id] = model
    return model


def replace_lc_object_types(type_annotation: typing.Any) -> typing.Any:
    """Recursively replaces all LangChain objects with a serialized representation.

    Args:
        type_annotation: The type annotation to replace.

    Returns:
        The type annotation with all LCObject types replaced.
    """
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    if args:
        if isinstance(args, (list, tuple)):
            new_args = [replace_lc_object_types(arg) for arg in args]

        if isinstance(origin, type):
            if origin is list:
                return typing.List[new_args[0]]
            elif origin is tuple:
                return typing.Tuple[tuple(new_args)]
            else:
                raise ValueError(f"Unknown origin type: {origin}")
        else:
            new_args = [replace_lc_object_types(arg) for arg in args]
            return origin[tuple(new_args)]

    if isinstance(type_annotation, type):
        if issubclass(type_annotation, Serializable):
            lc_id = type_annotation.get_lc_namespace() + [type_annotation.__name__]
            return _create_lc_object_validator(lc_id)

    return type_annotation
