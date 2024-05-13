"""Internal tracer to power the event stream API."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)
from uuid import UUID

from tenacity import RetryCallState
from typing_extensions import TypedDict

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.memory_stream import _MemoryStream

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


class RunInfo(TypedDict):
    """Information about a run."""

    name: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    run_type: str


class _AstreamEventHandler(AsyncCallbackHandler):
    """An implementation of an async callback handler for astream events."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the tracer."""
        super().__init__(*args, **kwargs)
        self.run_map: Dict[UUID, RunInfo] = {}
        """Map of run ID to run. Cleared on run end."""

        loop = asyncio.get_event_loop()
        memory_stream = _MemoryStream[StreamEvent](loop)
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()

    def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate over the receive stream."""
        return self.receive_stream.__aiter__()

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator
    ) -> AsyncIterator:
        """Tap the output aiter."""
        async for chunk in output:
            run_info = self.run_map.get(run_id)
            if run_info is None:
                raise AssertionError(f"Run ID {run_id} not found in run map.")
            await self.send_stream.send(
                {
                    "event": f"on_{run_info['run_type']}_stream",
                    "data": {"chunk": chunk},
                    "run_id": run_id,
                    "name": run_info["name"],
                    "tags": run_info["tags"],
                    "metadata": run_info["metadata"],
                }
            )
            yield chunk

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name,
            "run_type": "chat_model",
        }

        await self.send_stream.send(
            {
                "event": "on_chat_model_start",
                "data": {
                    "messages": messages,
                },
                "name": name,
                "tags": tags or [],
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name,
            "run_type": "llm",
        }

        await self.send_stream.send(
            {
                "event": "on_llm_start",
                "data": {
                    "prompts": prompts,
                },
                "name": name,
                "tags": tags or [],
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        run_info = self.run_map.get(run_id)
        await self.send_stream.send(
            {
                "event": "on_llm_new_token",
                "data": {
                    "token": token,
                    "chunk": chunk,
                },
                "run_id": run_id,
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            }
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()
        llm_run = self._get_run(run_id)
        retry_d: Dict[str, Any] = {
            "slept": retry_state.idle_for,
            "attempt": retry_state.attempt_number,
        }
        if retry_state.outcome is None:
            retry_d["outcome"] = "N/A"
        elif retry_state.outcome.failed:
            retry_d["outcome"] = "failed"
            exception = retry_state.outcome.exception()
            retry_d["exception"] = str(exception)
            retry_d["exception_type"] = exception.__class__.__name__
        else:
            retry_d["outcome"] = "success"
            retry_d["result"] = str(retry_state.outcome.result())
        llm_run.events.append(
            {
                "name": "retry",
                "time": datetime.now(timezone.utc),
                "kwargs": retry_d,
            },
        )
        return llm_run

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        run_info = self.run_map.pop(run_id)

        if run_info["run_type"] == "chat_model":
            event = "on_chat_model_end"
        elif run_info["run_type"] == "llm":
            event = "on_llm_end"
        else:
            raise ValueError(f"Unexpected run type: {run_info['run_type']}")

        await self.send_stream.send(
            {
                "event": event,
                "data": {
                    "response": response.dict(),
                },
                "run_id": run_id,
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            }
        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a chain run."""
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name,
            "run_type": "chain",
        }

        data = {}

        # Work-around Runnable core code not sending input in some
        # cases.
        if inputs != {"input": ""}:
            data["input"] = inputs

        await self.send_stream.send(
            {
                "event": "on_chain_start",
                "data": data,
                "name": name,
                "tags": tags or [],
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """End a trace for a chain run."""
        run_info = self.run_map.pop(run_id)

        await self.send_stream.send(
            {
                "event": "on_chain_end",
                "data": {
                    "input": inputs or {},
                    "output": outputs,
                },
                "run_id": run_id,
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            }
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a tool run."""
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name,
            "run_type": "tool",
        }

        await self.send_stream.send(
            {
                "event": "on_tool_start",
                "data": {
                    "input": inputs or {},
                },
                "name": name,
                "tags": tags or [],
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """End a trace for a tool run."""
        run_info = self.run_map.pop(run_id)
        await self.send_stream.send(
            {
                "event": "on_tool_end",
                "data": {
                    "output": output,
                },
                "run_id": run_id,
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            }
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when Retriever starts running."""
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name,
            "run_type": "tool",
        }

        await self.send_stream.send(
            {
                "event": "on_retriever_start",
                "data": {
                    "query": query,
                },
                "name": name,
                "tags": tags or [],
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    async def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Run when Retriever ends running."""
        run_info = self.run_map.pop(run_id)

        await self.send_stream.send(
            {
                "event": "on_retriever_end",
                "data": {
                    "documents": documents,
                },
                "run_id": run_id,
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            }
        )

    def __deepcopy__(self, memo: dict) -> _AstreamEventHandler:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> _AstreamEventHandler:
        """Copy the tracer."""
        return self


async def _event_stream_implementation(
    runnable: Runnable[Input, Output],
    input: Any,
    config: Optional[RunnableConfig] = None,
    *,
    stream: _AstreamEventHandler,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Implementation of astream_log for a given runnable.

    The implementation has been factored out (at least temporarily) as both
    astream_log and astream_events relies on it.
    """
    from langchain_core.callbacks.base import BaseCallbackManager
    from langchain_core.runnables import ensure_config

    # Assign the stream handler to the config
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [stream]
    elif isinstance(callbacks, list):
        config["callbacks"] = callbacks + [stream]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(stream, inherit=True)
        config["callbacks"] = callbacks
    else:
        raise ValueError(
            f"Unexpected type for callbacks: {callbacks}."
            "Expected None, list or AsyncCallbackManager."
        )

    # Call the runnable in streaming mode,
    # add each chunk to the output stream
    async def consume_astream() -> None:
        try:
            async for _ in runnable.astream(input, config, **kwargs):
                # All the content will be picked up
                pass
        finally:
            await stream.send_stream.aclose()

    # Start the runnable in a task, so we can start consuming output
    task = asyncio.create_task(consume_astream())
    try:
        async for log in stream:
            yield log
    finally:
        # Wait for the runnable to finish, if not cancelled (eg. by break)
        try:
            await task
        except asyncio.CancelledError:
            pass
