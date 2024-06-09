"""Internal tracer to power the event stream API."""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID, uuid4

from typing_extensions import NotRequired, TypedDict

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import (
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
from langchain_core.tracers.log_stream import LogEntry
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.utils.aiter import py_anext

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
    parent_run_id: Optional[UUID]


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
        # the entry corresponding to a given run id is cleaned
        # up when each corresponding run ends.
        self.run_map: Dict[UUID, RunInfo] = {}
        # The callback event that corresponds to the end of a parent run
        # may be invoked BEFORE the callback event that corresponds to the end
        # of a child run, which results in clean up of run_map.
        # So we keep track of the mapping between children and parent run IDs
        # in a separate container. This container is GCed when the tracer is GCed.
        self.parent_map: Dict[UUID, Optional[UUID]] = {}

        self.is_tapped: Dict[UUID, Any] = {}

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

    def _get_parent_ids(self, run_id: UUID) -> List[str]:
        """Get the parent IDs of a run (non-recursively) cast to strings."""
        parent_ids = []

        while parent_id := self.parent_map.get(run_id):
            str_parent_id = str(parent_id)
            if str_parent_id in parent_ids:
                raise AssertionError(
                    f"Parent ID {parent_id} is already in the parent_ids list. "
                    f"This should never happen."
                )
            parent_ids.append(str_parent_id)
            run_id = parent_id

        # Return the parent IDs in reverse order, so that the first
        # parent ID is the root and the last ID is the immediate parent.
        return parent_ids[::-1]

    def _send(self, event: StreamEvent, event_type: str) -> None:
        """Send an event to the stream."""
        if self.root_event_filter.include_event(event, event_type):
            self.send_stream.send_nowait(event)

    def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate over the receive stream."""
        return self.receive_stream.__aiter__()

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Tap the output aiter."""
        sentinel = object()
        # atomic check and set
        tap = self.is_tapped.setdefault(run_id, sentinel)
        # wait for first chunk
        first = await py_anext(output, default=sentinel)
        if first is sentinel:
            return
        # get run info
        run_info = self.run_map.get(run_id)
        if run_info is None:
            # run has finished, don't issue any stream events
            yield cast(T, first)
            return
        if tap is sentinel:
            # if we are the first to tap, issue stream events
            event: StreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast(T, first)
            # consume the rest of the output
            async for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
            # otherwise just pass through
            yield cast(T, first)
            # consume the rest of the output
            async for chunk in output:
                yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Tap the output aiter."""
        sentinel = object()
        # atomic check and set
        tap = self.is_tapped.setdefault(run_id, sentinel)
        # wait for first chunk
        first = next(output, sentinel)
        if first is sentinel:
            return
        # get run info
        run_info = self.run_map.get(run_id)
        if run_info is None:
            # run has finished, don't issue any stream events
            yield cast(T, first)
            return
        if tap is sentinel:
            # if we are the first to tap, issue stream events
            event: StreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast(T, first)
            # consume the rest of the output
            for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
            # otherwise just pass through
            yield cast(T, first)
            # consume the rest of the output
            for chunk in output:
                yield chunk

    def _write_run_start_info(
        self,
        run_id: UUID,
        *,
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        parent_run_id: Optional[UUID],
        name_: str,
        run_type: str,
        **kwargs: Any,
    ) -> None:
        """Update the run info."""
        info: RunInfo = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "parent_run_id": parent_run_id,
        }

        if "inputs" in kwargs:
            # Handle inputs in a special case to allow inputs to be an
            # optionally provided and distinguish between missing value
            # vs. None value.
            info["inputs"] = kwargs["inputs"]

        self.run_map[run_id] = info
        self.parent_map[run_id] = parent_run_id

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

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"messages": messages},
        )

        self._send(
            {
                "event": "on_chat_model_start",
                "data": {
                    "input": {"messages": messages},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
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

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"prompts": prompts},
        )

        self._send(
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
                "parent_ids": self._get_parent_ids(run_id),
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
        if self.is_tapped.get(run_id):
            return
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

        self._send(
            {
                "event": event,
                "data": {
                    "chunk": chunk_,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        run_info = self.run_map.pop(run_id)
        inputs_ = run_info["inputs"]

        generations: Union[List[List[GenerationChunk]], List[List[ChatGenerationChunk]]]
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

        self._send(
            {
                "event": event,
                "data": {"output": output, "input": inputs_},
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
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

        data: EventData = {}

        # Work-around Runnable core code not sending input in some
        # cases.
        if inputs != {"input": ""}:
            data["input"] = inputs
            kwargs["inputs"] = inputs

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type_,
            **kwargs,
        )

        self._send(
            {
                "event": f"on_{run_type_}_start",
                "data": data,
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
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

        self._send(
            {
                "event": event,
                "data": data,
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
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

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type="tool",
            inputs=inputs,
        )

        self._send(
            {
                "event": "on_tool_start",
                "data": {
                    "input": inputs or {},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
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

        self._send(
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
                "parent_ids": self._get_parent_ids(run_id),
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

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"query": query},
        )

        self._send(
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
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    async def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Run when Retriever ends running."""
        run_info = self.run_map.pop(run_id)

        self._send(
            {
                "event": "on_retriever_end",
                "data": {
                    "output": documents,
                    "input": run_info["inputs"],
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    def __deepcopy__(self, memo: dict) -> _AstreamEventsCallbackHandler:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> _AstreamEventsCallbackHandler:
        """Copy the tracer."""
        return self


async def _astream_events_implementation_v1(
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
    from langchain_core.runnables import ensure_config
    from langchain_core.runnables.utils import _RootEventFilter
    from langchain_core.tracers.log_stream import (
        LogStreamCallbackHandler,
        RunLog,
        _astream_log_implementation,
    )

    stream = LogStreamCallbackHandler(
        auto_close=False,
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
        _schema_format="streaming_events",
    )

    run_log = RunLog(state=None)  # type: ignore[arg-type]
    encountered_start_event = False

    _root_event_filter = _RootEventFilter(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

    config = ensure_config(config)
    root_tags = config.get("tags", [])
    root_metadata = config.get("metadata", {})
    root_name = config.get("run_name", runnable.get_name())

    # Ignoring mypy complaint about too many different union combinations
    # This arises because many of the argument types are unions
    async for log in _astream_log_implementation(  # type: ignore[misc]
        runnable,
        input,
        config=config,
        stream=stream,
        diff=True,
        with_streamed_output_list=True,
        **kwargs,
    ):
        run_log = run_log + log

        if not encountered_start_event:
            # Yield the start event for the root runnable.
            encountered_start_event = True
            state = run_log.state.copy()

            event = StreamEvent(
                event=f"on_{state['type']}_start",
                run_id=state["id"],
                name=root_name,
                tags=root_tags,
                metadata=root_metadata,
                data={
                    "input": input,
                },
                parent_ids=[],  # Not supported in v1
            )

            if _root_event_filter.include_event(event, state["type"]):
                yield event

        paths = {
            op["path"].split("/")[2]
            for op in log.ops
            if op["path"].startswith("/logs/")
        }
        # Elements in a set should be iterated in the same order
        # as they were inserted in modern python versions.
        for path in paths:
            data: EventData = {}
            log_entry: LogEntry = run_log.state["logs"][path]
            if log_entry["end_time"] is None:
                if log_entry["streamed_output"]:
                    event_type = "stream"
                else:
                    event_type = "start"
            else:
                event_type = "end"

            if event_type == "start":
                # Include the inputs with the start event if they are available.
                # Usually they will NOT be available for components that operate
                # on streams, since those components stream the input and
                # don't know its final value until the end of the stream.
                inputs = log_entry["inputs"]
                if inputs is not None:
                    data["input"] = inputs
                pass

            if event_type == "end":
                inputs = log_entry["inputs"]
                if inputs is not None:
                    data["input"] = inputs

                # None is a VALID output for an end event
                data["output"] = log_entry["final_output"]

            if event_type == "stream":
                num_chunks = len(log_entry["streamed_output"])
                if num_chunks != 1:
                    raise AssertionError(
                        f"Expected exactly one chunk of streamed output, "
                        f"got {num_chunks} instead. This is impossible. "
                        f"Encountered in: {log_entry['name']}"
                    )

                data = {"chunk": log_entry["streamed_output"][0]}
                # Clean up the stream, we don't need it anymore.
                # And this avoids duplicates as well!
                log_entry["streamed_output"] = []

            yield StreamEvent(
                event=f"on_{log_entry['type']}_{event_type}",
                name=log_entry["name"],
                run_id=log_entry["id"],
                tags=log_entry["tags"],
                metadata=log_entry["metadata"],
                data=data,
                parent_ids=[],  #  Not supported in v1
            )

        # Finally, we take care of the streaming output from the root chain
        # if there is any.
        state = run_log.state
        if state["streamed_output"]:
            num_chunks = len(state["streamed_output"])
            if num_chunks != 1:
                raise AssertionError(
                    f"Expected exactly one chunk of streamed output, "
                    f"got {num_chunks} instead. This is impossible. "
                    f"Encountered in: {state['name']}"
                )

            data = {"chunk": state["streamed_output"][0]}
            # Clean up the stream, we don't need it anymore.
            state["streamed_output"] = []

            event = StreamEvent(
                event=f"on_{state['type']}_stream",
                run_id=state["id"],
                tags=root_tags,
                metadata=root_metadata,
                name=root_name,
                data=data,
                parent_ids=[],  # Not supported in v1
            )
            if _root_event_filter.include_event(event, state["type"]):
                yield event

    state = run_log.state

    # Finally yield the end event for the root runnable.
    event = StreamEvent(
        event=f"on_{state['type']}_end",
        name=root_name,
        run_id=state["id"],
        tags=root_tags,
        metadata=root_metadata,
        data={
            "output": state["final_output"],
        },
        parent_ids=[],  # Not supported in v1
    )
    if _root_event_filter.include_event(event, state["type"]):
        yield event


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
    run_id = cast(UUID, config.setdefault("run_id", uuid4()))
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
            # if astream also calls tap_output_aiter this will be a no-op
            async for _ in event_streamer.tap_output_aiter(
                run_id, runnable.astream(input, config, **kwargs)
            ):
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
                # we dont include the input in the event since it's guaranteed
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
