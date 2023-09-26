import typing
from typing import Any, List, Sequence, Type, Union

from langchain.load.serializable import Serializable
from langchain.schema.runnable import RunnableConfig

try:
    from pydantic.v1 import BaseModel, Field, create_model, validator
except ImportError:
    from pydantic import BaseModel, Field, create_model, validator

from typing_extensions import TypedDict

InputValidator = Union[Type[BaseModel], type]
# The following langchain objects are considered to be safe to load.


def _project_runnable_config_type(keys: Sequence[str]) -> type(TypedDict):
    """Create a projection of the runnable config type.

    Args:
        keys: The keys to include in the projection.
    """
    subset_dict = {}
    for key in keys:
        if key in RunnableConfig.__annotations__:
            subset_dict[key] = RunnableConfig.__annotations__[key]
        else:
            raise AssertionError(f"Key {key} not in RunnableConfig.")

    return TypedDict("RunnableConfig", subset_dict, total=False)


def create_invoke_request_model(
    input_type: InputValidator,
    *,
    config_keys: Sequence[str] = (),
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    config_type = _project_runnable_config_type(config_keys)

    invoke_request_type = create_model(
        "InvokeRequest",
        input=(input_type, ...),
        config=(config_type, Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    invoke_request_type.update_forward_refs()
    return invoke_request_type


def create_batch_request_model(
    input_type: InputValidator,
    *,
    config_keys: Sequence[str] = (),
) -> Type[BaseModel]:
    """Create a pydantic model for the batch request."""
    config_type = _project_runnable_config_type(config_keys)
    batch_request_type = create_model(
        "BatchRequest",
        inputs=(List[input_type], ...),
        config=(Union[config_type, List[config_type]], Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    batch_request_type.update_forward_refs()
    return batch_request_type


def _create_lc_object_validator(expected_id: Sequence[str]) -> Type[BaseModel]:
    """Create a validator for lc objects."""
    expected_id = tuple(expected_id)

    class LCObject(BaseModel):
        """A model validator for lc objects."""

        id: typing.List[str]
        lc: Any
        type: str
        kwargs: Any

        @validator("id", allow_reuse=True)
        def validate_id_namespace(cls, id: Sequence[str]) -> None:
            """Validate that the LCObject is one of the allowed types."""
            if tuple(id) != expected_id:
                raise ValueError(f"LCObject id {id} is not allowed: {expected_id}")
            return id

    return LCObject


# PUBLIC API


def replace_lc_object_types(type_annotation: typing.Any) -> typing.Any:
    """Recursively replaces all types in a given type annotation."""
    if isinstance(type_annotation, typing._GenericAlias):
        # Handle generic types like List[int], Dict[str, int], etc.
        origin = typing.get_origin(type_annotation)
        args = [
            replace_lc_object_types(arg) for arg in typing.get_args(type_annotation)
        ]

        if origin is list:
            if args:
                return typing.List[args[0]]
            else:
                return typing.List
        elif origin is dict:
            # Special case for Dict in Python 3.8; use typing.Dict
            if args:
                return typing.Dict[args[0], args[1]]
            else:
                return typing.Dict
        elif origin is tuple:
            # Special case for Tuple in Python 3.8; use typing.Tuple
            if args:
                return typing.Tuple[args]
            else:
                return typing.Tuple
        else:
            raise NotImplementedError()
    elif isinstance(type_annotation, type) and issubclass(
        type_annotation, Serializable
    ):
        # Handle types that inherit from Serializable
        lc_id = type_annotation.get_lc_namespace() + [type_annotation.__name__]
        return _create_lc_object_validator(lc_id)
    else:
        # Return other types as-is
        return type_annotation
