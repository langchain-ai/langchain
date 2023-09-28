from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Sequence,
    Type,
    Union,
)

from langchain.load.dump import dumpd, dumps
from langchain.load.load import load
from langchain.schema.runnable import Runnable
from typing_extensions import Annotated

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
    create_runnable_config_model,
    create_stream_log_request_model,
    create_stream_request_model,
    replace_lc_object_types,
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


# PUBLIC API


def add_routes(
    app: Union[FastAPI, APIRouter],
    runnable: Runnable,
    *,
    path: str = "",
    input_type: Type = Any,
    config_keys: Sequence[str] = (),
) -> None:
    """Register the routes on the given FastAPI app or APIRouter.

    Args:
        app: The FastAPI app or APIRouter to which routes should be added.
        runnable: The runnable to wrap, must not be stateful.
        path: A path to prepend to all routes.
        input_type: Optional type to define a schema for the input part of the request.
            If not provided, any input that can be de-serialized with LangChain's
            serializer will be accepted.
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

    input_type = replace_lc_object_types(input_type)

    namespace = path or ""

    model_namespace = path.strip("/").replace("/", "_")

    config = create_runnable_config_model(model_namespace, config_keys)
    InvokeRequest = create_invoke_request_model(model_namespace, input_type, config)
    BatchRequest = create_batch_request_model(model_namespace, input_type, config)
    # Stream request is the same as invoke request, but with a different response type
    StreamRequest = create_stream_request_model(model_namespace, input_type, config)
    StreamLogRequest = create_stream_log_request_model(
        model_namespace, input_type, config
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
        input = load(request.dict()["input"])
        config = _project_dict(request.config, config_keys)
        output = await runnable.ainvoke(input, config=config, **request.kwargs)
        return InvokeResponse(output=dumpd(output))

    #
    @app.post(f"{namespace}/batch", response_model=BatchResponse)
    async def batch(request: Annotated[BatchRequest, BatchRequest]) -> BatchResponse:
        """Invoke the runnable with the given inputs and config."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        inputs = load(request.dict()["inputs"])
        if isinstance(request.config, list):
            config = [_project_dict(config, config_keys) for config in request.config]
        else:
            config = _project_dict(request.config, config_keys)
        output = await runnable.abatch(inputs, config=config, **request.kwargs)
        return BatchResponse(output=dumpd(output))

    @app.post(f"{namespace}/stream")
    async def stream(
        request: Annotated[StreamRequest, StreamRequest],
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input = load(request.dict()["input"])
        config = _project_dict(request.config, config_keys)

        async def _stream() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for chunk in runnable.astream(
                input,
                config=config,
                **request.kwargs,
            ):
                yield {"data": dumps(chunk), "event": "data"}
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
        input = load(request.dict()["input"])
        config = _project_dict(request.config, config_keys)

        async def _stream_log() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for run_log_patch in runnable.astream_log(
                input,
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
                    "data": dumps({"ops": run_log_patch.ops}),
                    "event": "data",
                }
            yield {"event": "end"}

        return EventSourceResponse(_stream_log())
