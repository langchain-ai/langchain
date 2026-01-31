"""Module contains typedefs that are used with `Runnable` objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type alias for stream event names used in astream_events
#
# When adding a new run_type, update these 3 locations:
# 1. StreamEventName Literal (here)
# 2. _RUN_TYPE_EVENTS mapping (below)
# 3. tests/unit_tests/runnables/test_schema.py (EXPECTED_EVENT_NAMES)
StreamEventName = Literal[
    # LLM (non-chat models)
    "on_llm_start",
    "on_llm_stream",
    "on_llm_end",
    "on_llm_error",
    # Chat Model
    "on_chat_model_start",
    "on_chat_model_stream",
    "on_chat_model_end",
    # Tool
    "on_tool_start",
    "on_tool_stream",
    "on_tool_end",
    "on_tool_error",
    # Chain
    "on_chain_start",
    "on_chain_stream",
    "on_chain_end",
    "on_chain_error",
    # Retriever
    "on_retriever_start",
    "on_retriever_stream",
    "on_retriever_end",
    "on_retriever_error",
    # Prompt
    "on_prompt_start",
    "on_prompt_end",
    # Parser
    "on_parser_start",
    "on_parser_stream",
    "on_parser_end",
    "on_parser_error",
    # Custom
    "on_custom_event",
]

# Type alias for event phases
StreamEventPhase = Literal["start", "stream", "end", "error"]

# Centralized mapping of run_type/phase to event names
# This ensures type safety and provides runtime validation
_RUN_TYPE_EVENTS: dict[str, dict[str, StreamEventName]] = {
    "llm": {
        "start": "on_llm_start",
        "stream": "on_llm_stream",
        "end": "on_llm_end",
        "error": "on_llm_error",
    },
    "chat_model": {
        "start": "on_chat_model_start",
        "stream": "on_chat_model_stream",
        "end": "on_chat_model_end",
    },
    "tool": {
        "start": "on_tool_start",
        "stream": "on_tool_stream",
        "end": "on_tool_end",
        "error": "on_tool_error",
    },
    "chain": {
        "start": "on_chain_start",
        "stream": "on_chain_stream",
        "end": "on_chain_end",
        "error": "on_chain_error",
    },
    "retriever": {
        "start": "on_retriever_start",
        "stream": "on_retriever_stream",
        "end": "on_retriever_end",
        "error": "on_retriever_error",
    },
    "prompt": {
        "start": "on_prompt_start",
        "end": "on_prompt_end",
    },
    "parser": {
        "start": "on_parser_start",
        "stream": "on_parser_stream",
        "end": "on_parser_end",
        "error": "on_parser_error",
    },
}


def get_event_name(run_type: str, phase: StreamEventPhase) -> StreamEventName:
    """Get the event name for a given run_type and phase.

    Args:
        run_type: The type of run (e.g., "llm", "chat_model", "tool", "chain").
        phase: The event phase ("start", "stream", "end", "error").

    Returns:
        The corresponding StreamEventName.

    Raises:
        ValueError: If the run_type/phase combination is not valid.
    """
    if run_type not in _RUN_TYPE_EVENTS:
        msg = (
            f"Unknown run_type: {run_type!r}. "
            f"Valid run_types: {list(_RUN_TYPE_EVENTS.keys())}"
        )
        raise ValueError(msg)

    phases = _RUN_TYPE_EVENTS[run_type]
    if phase not in phases:
        msg = (
            f"Phase {phase!r} is not valid for run_type {run_type!r}. "
            f"Valid phases: {list(phases.keys())}"
        )
        raise ValueError(msg)

    return phases[phase]


class EventData(TypedDict, total=False):
    """Data associated with a streaming event."""

    input: Any
    """The input passed to the `Runnable` that generated the event.

    Inputs will sometimes be available at the *START* of the `Runnable`, and
    sometimes at the *END* of the `Runnable`.

    If a `Runnable` is able to stream its inputs, then its input by definition
    won't be known until the *END* of the `Runnable` when it has finished streaming
    its inputs.
    """
    error: NotRequired[BaseException]
    """The error that occurred during the execution of the `Runnable`.

    This field is only available if the `Runnable` raised an exception.

    !!! version-added "Added in `langchain-core` 1.0.0"
    """
    output: Any
    """The output of the `Runnable` that generated the event.

    Outputs will only be available at the *END* of the `Runnable`.

    For most `Runnable` objects, this field can be inferred from the `chunk` field,
    though there might be some exceptions for special a cased `Runnable` (e.g., like
    chat models), which may return more information.
    """
    chunk: Any
    """A streaming chunk from the output that generated the event.

    chunks support addition in general, and adding them up should result
    in the output of the `Runnable` that generated the event.
    """
    tool_call_id: NotRequired[str | None]
    """The tool call ID associated with the tool execution.

    This field is available for the `on_tool_error` event and can be used to
    link errors to specific tool calls in stateless agent implementations.
    """


class BaseStreamEvent(TypedDict):
    """Streaming event.

    Schema of a streaming event which is produced from the `astream_events` method.

    Example:
        ```python
        from langchain_core.runnables import RunnableLambda


        async def reverse(s: str) -> str:
            return s[::-1]


        chain = RunnableLambda(func=reverse)

        events = [event async for event in chain.astream_events("hello")]

        # Will produce the following events
        # (where some fields have been omitted for brevity):
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
            {
                "data": {"output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
        ]
        ```
    """

    event: StreamEventName
    """Event names are of the format: `on_[runnable_type]_(start|stream|end)`.

    Runnable types are one of:

    - **llm** - used by non chat models
    - **chat_model** - used by chat models
    - **prompt** --  e.g., `ChatPromptTemplate`
    - **tool** -- from tools defined via `@tool` decorator or inheriting
        from `Tool`/`BaseTool`
    - **chain** - most `Runnable` objects are of this type

    Further, the events are categorized as one of:

    - **start** - when the `Runnable` starts
    - **stream** - when the `Runnable` is streaming
    - **end* - when the `Runnable` ends

    start, stream and end are associated with slightly different `data` payload.

    Please see the documentation for `EventData` for more details.
    """
    run_id: str
    """An randomly generated ID to keep track of the execution of the given `Runnable`.

    Each child `Runnable` that gets invoked as part of the execution of a parent
    `Runnable` is assigned its own unique ID.
    """
    tags: NotRequired[list[str]]
    """Tags associated with the `Runnable` that generated this event.

    Tags are always inherited from parent `Runnable` objects.

    Tags can either be bound to a `Runnable` using `.with_config({"tags":  ["hello"]})`
    or passed at run time using `.astream_events(..., {"tags": ["hello"]})`.
    """
    metadata: NotRequired[dict[str, Any]]
    """Metadata associated with the `Runnable` that generated this event.

    Metadata can either be bound to a `Runnable` using

        `.with_config({"metadata": { "foo": "bar" }})`

    or passed at run time using

        `.astream_events(..., {"metadata": {"foo": "bar"}})`.
    """

    parent_ids: Sequence[str]
    """A list of the parent IDs associated with this event.

    Root Events will have an empty list.

    For example, if a `Runnable` A calls `Runnable` B, then the event generated by
    `Runnable` B will have `Runnable` A's ID in the `parent_ids` field.

    The order of the parent IDs is from the root parent to the immediate parent.

    Only supported as of v2 of the astream events API. v1 will return an empty list.
    """


class StandardStreamEvent(BaseStreamEvent):
    """A standard stream event that follows LangChain convention for event data."""

    data: EventData
    """Event data.

    The contents of the event data depend on the event type.
    """
    name: str
    """The name of the `Runnable` that generated the event."""


class CustomStreamEvent(BaseStreamEvent):
    """Custom stream event created by the user."""

    # Overwrite the event field to be more specific.
    event: Literal["on_custom_event"]  # type: ignore[misc]
    """The event type."""
    name: str
    """User defined name for the event."""
    data: Any
    """The data associated with the event. Free form and can be anything."""


StreamEvent = StandardStreamEvent | CustomStreamEvent
