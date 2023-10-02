from inspect import isclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Type,
    Union,
)

from langchain.schema.runnable import Runnable
from typing_extensions import Annotated

try:
    from pydantic.v1 import BaseModel, create_model
except ImportError:
    from pydantic import BaseModel, create_model

from langserve.serialization import simple_dumpd, simple_dumps
from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
    create_runnable_config_model,
    create_stream_log_request_model,
    create_stream_request_model,
)

try:
    from fastapi import APIRouter, FastAPI
except ImportError:
    # [server] extra not installed
    APIRouter = FastAPI = Any


def _project_dict(d: Mapping, keys: Sequence[str]) -> Dict[str, Any]:
    """Project the given keys from the given dict."""
    return {k: d[k] for k in keys if k in d}


class InvokeResponse(BaseModel):
    """Response from invoking a runnable.

    A container is used to allow adding additional fields in the future.
    """

    output: Any
    """The output of the runnable.

    An object that can be serialized to JSON using LangChain serialization.
    """


class BatchResponse(BaseModel):
    """Response from batch invoking runnables.

    A container is used to allow adding additional fields in the future.
    """

    output: List[Any]
    """The output of the runnable.

    An object that can be serialized to JSON using LangChain serialization.
    """


def _unpack_input(validated_model: BaseModel) -> Any:
    """Unpack the decoded input from the validated model."""
    if hasattr(validated_model, "__root__"):
        return validated_model.__root__
    else:
        return validated_model


_MODEL_REGISTRY = {}


def _resolve_input_type(input_type: Union[Type, BaseModel]) -> BaseModel:
    if isclass(input_type) and issubclass(input_type, BaseModel):
        input_type_ = input_type
    else:
        input_type_ = create_model("Input", __root__=(input_type, ...))

    hash_ = input_type_.schema_json()

    if hash_ not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[hash_] = input_type_

    return _MODEL_REGISTRY[hash_]


# PUBLIC API


def add_routes(
    app: Union[FastAPI, APIRouter],
    runnable: Runnable,
    *,
    path: str = "",
    input_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    config_keys: Sequence[str] = (),
) -> None:
    """Register the routes on the given FastAPI app or APIRouter.

    Args:
        app: The FastAPI app or APIRouter to which routes should be added.
        runnable: The runnable to wrap, must not be stateful.
        path: A path to prepend to all routes.
        input_type: type to use for input validation.
            Default is "auto" which will use the InputType of the runnable.
            User is free to provide a custom type annotation.
        config_keys: list of config keys that will be accepted, by default
                     no config keys are accepted.
    """
    try:
        from sse_starlette import EventSourceResponse
    except ImportError:
        raise ImportError(
            "sse_starlette must be installed to implement the stream and "
            "stream_log endpoints. "
            "Use `pip install sse_starlette` to install."
        )

    if input_type == "auto":
        input_type_ = _resolve_input_type(runnable.input_schema)
    else:
        input_type_ = _resolve_input_type(input_type)

    namespace = path or ""

    model_namespace = path.strip("/").replace("/", "_")

    config = create_runnable_config_model(model_namespace, config_keys)
    InvokeRequest = create_invoke_request_model(model_namespace, input_type_, config)
    BatchRequest = create_batch_request_model(model_namespace, input_type_, config)
    # Stream request is the same as invoke request, but with a different response type
    StreamRequest = create_stream_request_model(model_namespace, input_type_, config)
    StreamLogRequest = create_stream_log_request_model(
        model_namespace, input_type_, config
    )

    @app.post(
        f"{namespace}/invoke",
        response_model=InvokeResponse,
    )
    async def invoke(
        request: Annotated[InvokeRequest, InvokeRequest]
    ) -> InvokeResponse:
        """Invoke the runnable with the given input and config."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        config = _project_dict(request.config, config_keys)
        output = await runnable.ainvoke(
            _unpack_input(request.input), config=config, **request.kwargs
        )

        return InvokeResponse(output=simple_dumpd(output))

    #
    @app.post(f"{namespace}/batch", response_model=BatchResponse)
    async def batch(request: Annotated[BatchRequest, BatchRequest]) -> BatchResponse:
        """Invoke the runnable with the given inputs and config."""
        if isinstance(request.config, list):
            config = [_project_dict(config, config_keys) for config in request.config]
        else:
            config = _project_dict(request.config, config_keys)
        inputs = [_unpack_input(input_) for input_ in request.inputs]
        output = await runnable.abatch(inputs, config=config, **request.kwargs)

        return BatchResponse(output=simple_dumpd(output))

    @app.post(f"{namespace}/stream")
    async def stream(
        request: Annotated[StreamRequest, StreamRequest],
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input_ = _unpack_input(request.input)
        config = _project_dict(request.config, config_keys)

        async def _stream() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for chunk in runnable.astream(
                input_,
                config=config,
                **request.kwargs,
            ):
                yield {"data": simple_dumps(chunk), "event": "data"}
            yield {"event": "end"}

        return EventSourceResponse(_stream())

    @app.post(f"{namespace}/stream_log")
    async def stream_log(
        request: Annotated[StreamLogRequest, StreamLogRequest],
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input_ = _unpack_input(request.input)
        config = _project_dict(request.config, config_keys)

        async def _stream_log() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for run_log_patch in runnable.astream_log(
                input_,
                config=config,
                include_names=request.include_names,
                include_types=request.include_types,
                include_tags=request.include_tags,
                exclude_names=request.exclude_names,
                exclude_types=request.exclude_types,
                exclude_tags=request.exclude_tags,
                **request.kwargs,
            ):
                # Temporary adapter
                yield {
                    "data": simple_dumps({"ops": run_log_patch.ops}),
                    "event": "data",
                }
            yield {"event": "end"}

        return EventSourceResponse(_stream_log())
