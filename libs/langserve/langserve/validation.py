from typing import List, Optional, Sequence, Type, Union

from langchain.schema.runnable import RunnableConfig

try:
    from pydantic.v1 import BaseModel, Field, create_model
except ImportError:
    from pydantic import BaseModel, Field, create_model

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
