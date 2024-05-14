"""Internal tracer to power the event stream API."""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

from typing_extensions import NotRequired, TypedDict

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
    Input,
    Output,
    _RootEventFilter,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.memory_stream import _MemoryStream

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnableConfig

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


T = TypeVar("T")


class _AstreamEventsCallbackHandler(AsyncCallbackHandler, _StreamingCallbackHandler):
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
        # Map of run ID to run info.
        self.run_map: Dict[UUID, RunInfo] = {}

        # Filter which events will be sent over the queue.
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
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
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

        chunk_: Union[GenerationChunk, BaseMessageChunk]

        if run_info is None:
            raise AssertionError(f"Run ID {run_id} not found in run map.")
        if run_info["run_type"] == "chat_model":
            event = "on_chat_model_stream"

            if chunk is None:
                chunk_ = AIMessageChunk(content=token)
            else:
                chunk_ = cast(ChatGenerationChunk, chunk).message

        elif run_info["run_type"] == "llm":
            event = "on_llm_stream"
            if chunk is None:
                chunk_ = GenerationChunk(text=token)
            else:
                chunk_ = cast(GenerationChunk, chunk)
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

        generations: Union[List[List[GenerationChunk]], List[List[ChatGeneration]]]
        output: Union[dict, BaseMessage] = {}

        if run_info["run_type"] == "chat_model":
            generations = cast(List[List[ChatGenerationChunk]], response.generations)
            for gen in generations:
                if output != {}:
                    break
                for chunk in gen:
                    output = chunk.message
                    break

            event = "on_chat_model_end"
        elif run_info["run_type"] == "llm":
            generations = cast(List[List[GenerationChunk]], response.generations)
            output = {
                "generations": [
                    [
                        {
                            "text": chunk.text,
                            "generation_info": chunk.generation_info,
                            "type": chunk.type,
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
        run_info: RunInfo = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type_,
        }

        data: EventData = {}

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

        data: EventData = {
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

    def __deepcopy__(self, memo: dict) -> _AstreamEventsCallbackHandler:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> _AstreamEventsCallbackHandler:
        """Copy the tracer."""
        return self


async def _astream_events_implementation_v2(
    runnable: Runnable[Input, Output],
    input: Any,
    config: Optional[RunnableConfig] = None,
    *,
    include_names: Optional[Sequence[str]] = None,
    include_types: Optional[Sequence[str]] = None,
    include_tags: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
    exclude_types: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Implementation of the astream events API for V2 runnables."""
    from langchain_core.callbacks.base import BaseCallbackManager
    from langchain_core.runnables import ensure_config

    event_streamer = _AstreamEventsCallbackHandler(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

    # Assign the stream handler to the config
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [event_streamer]
    elif isinstance(callbacks, list):
        config["callbacks"] = callbacks + [event_streamer]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(event_streamer, inherit=True)
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
            await event_streamer.send_stream.aclose()

    # Start the runnable in a task, so we can start consuming output
    task = asyncio.create_task(consume_astream())

    first_event_sent = False
    first_event_run_id = None

    try:
        async for event in event_streamer:
            if not first_event_sent:
                first_event_sent = True
                # This is a work-around an issue where the inputs into the
                # chain are not available until the entire input is consumed.
                # As a temporary solution, we'll modify the input to be the input
                # that was passed into the chain.
                event["data"]["input"] = input
                first_event_run_id = event["run_id"]
                yield event
                continue

            if event["run_id"] == first_event_run_id and event["event"].endswith(
                "_end"
            ):
                # If it's the end event corresponding to the root runnable
                # we want to include the input in the event since it's guaranteed
                # to be included in the first event.
                if "input" in event["data"]:
                    del event["data"]["input"]

            yield event
    finally:
        # Wait for the runnable to finish, if not cancelled (eg. by break)
        try:
            await task
        except asyncio.CancelledError:
            pass
