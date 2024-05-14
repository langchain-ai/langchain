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
    cast,
)
from uuid import UUID

from tenacity import RetryCallState
from typing_extensions import NotRequired, TypedDict

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import (
    _RootEventFilter,
)
from langchain_core.tracers.memory_stream import _MemoryStream

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RunInfo(TypedDict):
    """Information about a run."""

    name: str
    tags: List[str]
    metadata: Dict[str, Any]
    run_type: str
    inputs: NotRequired[Any]


def _assign_name(name: Optional[str], serialized: Dict[str, Any]) -> str:
    """Assign a name to a run."""
    if name is not None:
        return name
    if "name" in serialized:
        return serialized["name"]
    elif "id" in serialized:
        return serialized["id"][-1]
    return "Unnamed"


class _AstreamEventHandler(AsyncCallbackHandler):
    """An implementation of an async callback handler for astream events."""

    def __init__(
        self,
        *args: Any,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tracer."""
        super().__init__(*args, **kwargs)
        self.run_map: Dict[UUID, RunInfo] = {}
        """Map of run ID to run. Cleared on run end."""

        self.root_event_filter = _RootEventFilter(
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
        )

        loop = asyncio.get_event_loop()
        memory_stream = _MemoryStream[StreamEvent](loop)
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()

    async def _send(self, event: StreamEvent, event_type: str) -> None:
        """Send an event to the stream."""
        if self.root_event_filter.include_event(event, event_type):
            await self.send_stream.send(event)

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
            await self._send(
                {
                    "event": f"on_{run_info['run_type']}_stream",
                    "data": {"chunk": chunk},
                    "run_id": str(run_id),
                    "name": run_info["name"],
                    "tags": run_info["tags"],
                    "metadata": run_info["metadata"],
                },
                run_info["run_type"],
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
        name_ = _assign_name(name, serialized)
        run_type = "chat_model"
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "inputs": {"messages": messages},
        }

        await self._send(
            {
                "event": "on_chat_model_start",
                "data": {
                    "input": {"messages": messages},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
            },
            run_type,
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
        name_ = _assign_name(name, serialized)
        run_type = "llm"
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "inputs": {"prompts": prompts},
        }

        await self._send(
            {
                "event": "on_llm_start",
                "data": {
                    "input": {
                        "prompts": prompts,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
            },
            run_type,
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
        if run_info["run_type"] == "chat_model":
            event = "on_chat_model_stream"
            chunk = cast(ChatGenerationChunk, chunk)
            chunk_ = chunk.message
        elif run_info["run_type"] == "llm":
            event = "on_llm_stream"
            chunk = cast(GenerationChunk, chunk)
            chunk_ = chunk
        else:
            raise ValueError(f"Unexpected run type: {run_info['run_type']}")

        await self._send(
            {
                "event": event,
                "data": {
                    "chunk": chunk_,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            },
            run_info["run_type"],
        )

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        run_info = self.run_map.pop(run_id)
        inputs_ = run_info["inputs"]

        if run_info["run_type"] == "chat_model":
            generations = cast(List[List[ChatGeneration]], response.generations)
            output = {
                "generations": [
                    [
                        {
                            "message": chunk.message,
                            "generation_info": chunk.generation_info,
                            "text": chunk.text,
                            "type": "ChatGenerationChunk",
                        }
                        for chunk in gen
                    ]
                    for gen in generations
                ],
                "llm_output": response.llm_output,
            }

            event = "on_chat_model_end"
        elif run_info["run_type"] == "llm":
            generations = cast(List[List[GenerationChunk]], response.generations)
            output = {
                "generations": [
                    [
                        {
                            "text": chunk.text,
                            "generation_info": chunk.generation_info,
                            "type": "Generation",
                        }
                        for chunk in gen
                    ]
                    for gen in generations
                ],
                "llm_output": response.llm_output,
            }
            event = "on_llm_end"
        else:
            raise ValueError(f"Unexpected run type: {run_info['run_type']}")

        await self._send(
            {
                "event": event,
                "data": {"output": output, "input": inputs_},
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            },
            run_info["run_type"],
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Run on retry."""
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
        name_ = _assign_name(name, serialized)
        run_type_ = run_type or "chain"
        run_info = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type_,
        }

        data = {}

        if inputs != {"input": ""}:
            data["input"] = inputs
            run_info["inputs"] = inputs

        self.run_map[run_id] = run_info

        # Work-around Runnable core code not sending input in some
        # cases.

        await self._send(
            {
                "event": f"on_{run_type_}_start",
                "data": data,
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
            },
            run_type_,
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
        run_type = run_info["run_type"]

        event = f"on_{run_type}_end"

        inputs = inputs or run_info.get("inputs") or {}

        data = {
            "output": outputs,
            "input": inputs,
        }

        await self._send(
            {
                "event": event,
                "data": data,
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            },
            run_type,
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
        name_ = _assign_name(name, serialized)
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": "tool",
            "inputs": inputs,
        }

        await self._send(
            {
                "event": "on_tool_start",
                "data": {
                    "input": inputs or {},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
            },
            "tool",
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """End a trace for a tool run."""
        run_info = self.run_map.pop(run_id)
        if "inputs" not in run_info:
            raise AssertionError(
                f"Run ID {run_id} is a tool call and is expected to have "
                f"inputs associated with it."
            )
        inputs = run_info["inputs"]

        await self._send(
            {
                "event": "on_tool_end",
                "data": {
                    "output": output,
                    "input": inputs,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            },
            "tool",
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
        name_ = _assign_name(name, serialized)
        run_type = "retriever"
        self.run_map[run_id] = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "inputs": {"query": query},
        }

        await self._send(
            {
                "event": "on_retriever_start",
                "data": {
                    "input": {
                        "query": query,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
            },
            run_type,
        )

    async def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Run when Retriever ends running."""
        run_info = self.run_map.pop(run_id)

        await self._send(
            {
                "event": "on_retriever_end",
                "data": {
                    "output": {"documents": documents},
                    "input": run_info["inputs"],
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
            },
            run_info["run_type"],
        )

    def __deepcopy__(self, memo: dict) -> _AstreamEventHandler:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> _AstreamEventHandler:
        """Copy the tracer."""
        return self
