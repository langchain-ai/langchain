from typing import (
    TYPE_CHECKING,
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

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
    replace_lc_object_types,
)

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI
else:
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
    from sse_starlette import EventSourceResponse

    input_type = replace_lc_object_types(input_type)

    namespace = path or ""

    InvokeRequest = create_invoke_request_model(input_type, config_keys=config_keys)
    BatchRequest = create_batch_request_model(input_type, config_keys=config_keys)
    # Stream request is the same as invoke request, but with a different response type
    StreamRequest = create_invoke_request_model(input_type, config_keys=config_keys)

    @app.post(f"{namespace}/invoke", response_model=InvokeResponse)
    async def invoke(request: InvokeRequest) -> InvokeResponse:
        """Invoke the runnable with the given input and config."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input = load(request.dict()["input"])
        config = _project_dict(request.config, config_keys)
        output = await runnable.ainvoke(input, config=config, **request.kwargs)
        return InvokeResponse(output=dumpd(output))

    @app.post(f"{namespace}/batch", response_model=BatchResponse)
    async def batch(request: BatchRequest) -> BatchResponse:
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
    async def stream(request: StreamRequest) -> EventSourceResponse:
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
