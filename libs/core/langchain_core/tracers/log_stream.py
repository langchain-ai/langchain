"""Tracer that streams run logs to a stream."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
from collections import defaultdict
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    overload,
)

import jsonpatch  # type: ignore[import-untyped]
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from uuid import UUID

    from langchain_core.runnables import Runnable
    from langchain_core.runnables.utils import Input, Output
    from langchain_core.tracers.schemas import Run


class LogEntry(TypedDict):
    """A single entry in the run log."""

    id: str
    """ID of the sub-run."""
    name: str
    """Name of the object being run."""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc."""
    tags: list[str]
    """List of tags for the run."""
    metadata: dict[str, Any]
    """Key-value pairs of metadata for the run."""
    start_time: str
    """ISO-8601 timestamp of when the run started."""

    streamed_output_str: list[str]
    """List of LLM tokens streamed by this run, if applicable."""
    streamed_output: list[Any]
    """List of output chunks streamed by this run, if available."""
    inputs: NotRequired[Any | None]
    """Inputs to this run. Not available currently via astream_log."""
    final_output: Any | None
    """Final output of this run.

    Only available after the run has finished successfully."""
    end_time: str | None
    """ISO-8601 timestamp of when the run ended.
    Only available after the run has finished."""


class RunState(TypedDict):
    """State of the run."""

    id: str
    """ID of the run."""
    streamed_output: list[Any]
    """List of output chunks streamed by Runnable.stream()"""
    final_output: Any | None
    """Final output of the run, usually the result of aggregating (`+`) streamed_output.
    Updated throughout the run when supported by the Runnable."""

    name: str
    """Name of the object being run."""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc."""

    # Do we want tags/metadata on the root run? Client kinda knows it in most situations
    # tags: list[str]

    logs: dict[str, LogEntry]
    """Map of run names to sub-runs. If filters were supplied, this list will
    contain only the runs that matched the filters."""


class RunLogPatch:
    """Patch to the run log."""

    ops: list[dict[str, Any]]
    """List of JSONPatch operations, which describe how to create the run state
    from an empty dict. This is the minimal representation of the log, designed to
    be serialized as JSON and sent over the wire to reconstruct the log on the other
    side. Reconstruction of the state can be done with any JSONPatch-compliant library,
    see https://jsonpatch.com for more information."""

    def __init__(self, *ops: dict[str, Any]) -> None:
        """Create a RunLogPatch.

        Args:
            *ops: The operations to apply to the state.
        """
        self.ops = list(ops)

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """Combine two `RunLogPatch` instances.

        Args:
            other: The other `RunLogPatch` to combine with.

        Raises:
            TypeError: If the other object is not a `RunLogPatch`.

        Returns:
            A new `RunLog` representing the combination of the two.
        """
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(None, copy.deepcopy(ops))
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        # 1:-1 to get rid of the [] around the list
        return f"RunLogPatch({pformat(self.ops)[1:-1]})"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunLogPatch) and self.ops == other.ops

    __hash__ = None  # type: ignore[assignment]


class RunLog(RunLogPatch):
    """Run log."""

    state: RunState
    """Current state of the log, obtained from applying all ops in sequence."""

    def __init__(self, *ops: dict[str, Any], state: RunState) -> None:
        """Create a RunLog.

        Args:
            *ops: The operations to apply to the state.
            state: The initial state of the run log.
        """
        super().__init__(*ops)
        self.state = state

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """Combine two `RunLog`s.

        Args:
            other: The other `RunLog` or `RunLogPatch` to combine with.

        Raises:
            TypeError: If the other object is not a `RunLog` or `RunLogPatch`.

        Returns:
            A new `RunLog` representing the combination of the two.
        """
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(self.state, other.ops)
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        return f"RunLog({pformat(self.state)})"

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two `RunLog`s are equal.

        Args:
            other: The other `RunLog` to compare to.

        Returns:
            `True` if the `RunLog`s are equal, `False` otherwise.
        """
        # First compare that the state is the same
        if not isinstance(other, RunLog):
            return False
        if self.state != other.state:
            return False
        # Then compare that the ops are the same
        return super().__eq__(other)

    __hash__ = None


T = TypeVar("T")


class LogStreamCallbackHandler(BaseTracer, _StreamingCallbackHandler):
    """Tracer that streams run logs to a stream."""

    def __init__(
        self,
        *,
        auto_close: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        # Schema format is for internal use only.
        _schema_format: Literal["original", "streaming_events"] = "streaming_events",
    ) -> None:
        """A tracer that streams run logs to a stream.

        Args:
            auto_close: Whether to close the stream when the root run finishes.
            include_names: Only include runs from Runnables with matching names.
            include_types: Only include runs from Runnables with matching types.
            include_tags: Only include runs from Runnables with matching tags.
            exclude_names: Exclude runs from Runnables with matching names.
            exclude_types: Exclude runs from Runnables with matching types.
            exclude_tags: Exclude runs from Runnables with matching tags.
            _schema_format: Primarily changes how the inputs and outputs are
                handled.

                **For internal use only. This API will change.**

                - 'original' is the format used by all current tracers.
                  This format is slightly inconsistent with respect to inputs
                  and outputs.
                - 'streaming_events' is used for supporting streaming events,
                  for internal usage. It will likely change in the future, or
                  be deprecated entirely in favor of a dedicated async tracer
                  for streaming events.

        Raises:
            ValueError: If an invalid schema format is provided (internal use only).
        """
        if _schema_format not in {"original", "streaming_events"}:
            msg = (
                f"Invalid schema format: {_schema_format}. "
                f"Expected one of 'original', 'streaming_events'."
            )
            raise ValueError(msg)
        super().__init__(_schema_format=_schema_format)

        self.auto_close = auto_close
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        memory_stream = _MemoryStream[RunLogPatch](loop)
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()
        self._key_map_by_run_id: dict[UUID, str] = {}
        self._counter_map_by_name: dict[str, int] = defaultdict(int)
        self.root_id: UUID | None = None

    def __aiter__(self) -> AsyncIterator[RunLogPatch]:
        """Iterate over the stream of run logs.

        Returns:
            An async iterator over the run log patches.
        """
        return self.receive_stream.__aiter__()

    def send(self, *ops: dict[str, Any]) -> bool:
        """Send a patch to the stream, return False if the stream is closed.

        Args:
            *ops: The operations to send to the stream.

        Returns:
            `True` if the patch was sent successfully, False if the stream is closed.
        """
        # We will likely want to wrap this in try / except at some point
        # to handle exceptions that might arise at run time.
        # For now we'll let the exception bubble up, and always return
        # True on the happy path.
        self.send_stream.send_nowait(RunLogPatch(*ops))
        return True

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Tap an output async iterator to stream its values to the log.

        Args:
            run_id: The ID of the run.
            output: The output async iterator.

        Yields:
            The output value.
        """
        async for chunk in output:
            # root run is handled in .astream_log()
            # if we can't find the run silently ignore
            # eg. because this run wasn't included in the log
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Tap an output async iterator to stream its values to the log.

        Args:
            run_id: The ID of the run.
            output: The output iterator.

        Yields:
            The output value.
        """
        for chunk in output:
            # root run is handled in .astream_log()
            # if we can't find the run silently ignore
            # eg. because this run wasn't included in the log
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def include_run(self, run: Run) -> bool:
        """Check if a Run should be included in the log.

        Args:
            run: The Run to check.

        Returns:
            `True` if the run should be included, `False` otherwise.
        """
        if run.id == self.root_id:
            return False

        run_tags = run.tags or []

        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        if self.include_names is not None:
            include = include or run.name in self.include_names
        if self.include_types is not None:
            include = include or run.run_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in run_tags)

        if self.exclude_names is not None:
            include = include and run.name not in self.exclude_names
        if self.exclude_types is not None:
            include = include and run.run_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(tag not in self.exclude_tags for tag in run_tags)

        return include

    def _persist_run(self, run: Run) -> None:
        # This is a legacy method only called once for an entire run tree
        # therefore not useful here
        pass

    def _on_run_create(self, run: Run) -> None:
        """Start a run."""
        if self.root_id is None:
            self.root_id = run.id
            if not self.send(
                {
                    "op": "replace",
                    "path": "",
                    "value": RunState(
                        id=str(run.id),
                        streamed_output=[],
                        final_output=None,
                        logs={},
                        name=run.name,
                        type=run.run_type,
                    ),
                }
            ):
                return

        if not self.include_run(run):
            return

        # Determine previous index, increment by 1
        with self.lock:
            self._counter_map_by_name[run.name] += 1
            count = self._counter_map_by_name[run.name]
            self._key_map_by_run_id[run.id] = (
                run.name if count == 1 else f"{run.name}:{count}"
            )

        entry = LogEntry(
            id=str(run.id),
            name=run.name,
            type=run.run_type,
            tags=run.tags or [],
            metadata=(run.extra or {}).get("metadata", {}),
            start_time=run.start_time.isoformat(timespec="milliseconds"),
            streamed_output=[],
            streamed_output_str=[],
            final_output=None,
            end_time=None,
        )

        if self._schema_format == "streaming_events":
            # If using streaming events let's add inputs as well
            entry["inputs"] = _get_standardized_inputs(run, self._schema_format)

        # Add the run to the stream
        self.send(
            {
                "op": "add",
                "path": f"/logs/{self._key_map_by_run_id[run.id]}",
                "value": entry,
            }
        )

    def _on_run_update(self, run: Run) -> None:
        """Finish a run."""
        try:
            index = self._key_map_by_run_id.get(run.id)

            if index is None:
                return

            ops = []

            if self._schema_format == "streaming_events":
                ops.append(
                    {
                        "op": "replace",
                        "path": f"/logs/{index}/inputs",
                        "value": _get_standardized_inputs(run, self._schema_format),
                    }
                )

            ops.extend(
                [
                    # Replace 'inputs' with final inputs
                    # This is needed because in many cases the inputs are not
                    # known until after the run is finished and the entire
                    # input stream has been processed by the runnable.
                    {
                        "op": "add",
                        "path": f"/logs/{index}/final_output",
                        # to undo the dumpd done by some runnables / tracer / etc
                        "value": _get_standardized_outputs(run, self._schema_format),
                    },
                    {
                        "op": "add",
                        "path": f"/logs/{index}/end_time",
                        "value": run.end_time.isoformat(timespec="milliseconds")
                        if run.end_time is not None
                        else None,
                    },
                ]
            )

            self.send(*ops)
        finally:
            if run.id == self.root_id and self.auto_close:
                self.send_stream.close()

    def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> None:
        """Process new LLM token."""
        index = self._key_map_by_run_id.get(run.id)

        if index is None:
            return

        self.send(
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output_str/-",
                "value": token,
            },
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output/-",
                "value": chunk.message
                if isinstance(chunk, ChatGenerationChunk)
                else token,
            },
        )


def _get_standardized_inputs(
    run: Run, schema_format: Literal["original", "streaming_events"]
) -> dict[str, Any] | None:
    """Extract standardized inputs from a run.

    Standardizes the inputs based on the type of the runnable used.

    Args:
        run: Run object
        schema_format: The schema format to use.

    Returns:
        Valid inputs are only dict. By conventions, inputs always represented
        invocation using named arguments.
        None means that the input is not yet known!
    """
    if schema_format == "original":
        msg = (
            "Do not assign inputs with original schema drop the key for now."
            "When inputs are added to astream_log they should be added with "
            "standardized schema for streaming events."
        )
        raise NotImplementedError(msg)

    inputs = load(run.inputs)

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return inputs

    # new style chains
    # These nest an additional 'input' key inside the 'inputs' to make sure
    # the input is always a dict. We need to unpack and user the inner value.
    inputs = inputs["input"]
    # We should try to fix this in Runnables and callbacks/tracers
    # Runnables should be using a None type here not a placeholder
    # dict.
    if inputs == {"input": ""}:  # Workaround for Runnables not using None
        # The input is not known, so we don't assign data['input']
        return None
    return inputs


def _get_standardized_outputs(
    run: Run, schema_format: Literal["original", "streaming_events", "original+chat"]
) -> Any | None:
    """Extract standardized output from a run.

    Standardizes the outputs based on the type of the runnable used.

    Args:
        run: the run object.
        schema_format: The schema format to use.

    Returns:
        An output if returned, otherwise a None
    """
    outputs = load(run.outputs)
    if schema_format == "original":
        if run.run_type == "prompt" and "output" in outputs:
            # These were previously dumped before the tracer.
            # Now we needn't do anything to them.
            return outputs["output"]
        # Return the old schema, without standardizing anything
        return outputs

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return outputs

    if isinstance(outputs, dict):
        return outputs.get("output", None)

    return None


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[True] = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch]: ...


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[False],
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLog]: ...


async def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
    """Implementation of astream_log for a given runnable.

    The implementation has been factored out (at least temporarily) as both
    astream_log and astream_events relies on it.

    Args:
        runnable: The runnable to run in streaming mode.
        value: The input to the runnable.
        config: The config to pass to the runnable.
        stream: The stream to send the run logs to.
        diff: Whether to yield run log patches (True) or full run logs (False).
        with_streamed_output_list: Whether to include a list of all streamed
            outputs in each patch. If `False`, only the final output will be included
            in the patches.
        **kwargs: Additional keyword arguments to pass to the runnable.

    Raises:
        ValueError: If the callbacks in the config are of an unexpected type.

    Yields:
        The run log patches or states, depending on the value of `diff`.
    """
    # Assign the stream handler to the config
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [stream]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, stream]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(stream, inherit=True)
        config["callbacks"] = callbacks
    else:
        msg = (
            f"Unexpected type for callbacks: {callbacks}."
            "Expected None, list or AsyncCallbackManager."
        )
        raise ValueError(msg)

    # Call the runnable in streaming mode,
    # add each chunk to the output stream
    async def consume_astream() -> None:
        try:
            prev_final_output: Output | None = None
            final_output: Output | None = None

            async for chunk in runnable.astream(value, config, **kwargs):
                prev_final_output = final_output
                if final_output is None:
                    final_output = chunk
                else:
                    try:
                        final_output = final_output + chunk  # type: ignore[operator]
                    except TypeError:
                        prev_final_output = None
                        final_output = chunk
                patches: list[dict[str, Any]] = []
                if with_streamed_output_list:
                    patches.append(
                        {
                            "op": "add",
                            "path": "/streamed_output/-",
                            # chunk cannot be shared between
                            # streamed_output and final_output
                            # otherwise jsonpatch.apply will
                            # modify both
                            "value": copy.deepcopy(chunk),
                        }
                    )
                patches.extend(
                    {**op, "path": f"/final_output{op['path']}"}
                    for op in jsonpatch.JsonPatch.from_diff(
                        prev_final_output, final_output, dumps=dumps
                    )
                )
                await stream.send_stream.send(RunLogPatch(*patches))
        finally:
            await stream.send_stream.aclose()

    # Start the runnable in a task, so we can start consuming output
    task = asyncio.create_task(consume_astream())
    try:
        # Yield each chunk from the output stream
        if diff:
            async for log in stream:
                yield log
        else:
            state = RunLog(state=None)  # type: ignore[arg-type]
            async for log in stream:
                state += log
                yield state
    finally:
        # Wait for the runnable to finish, if not cancelled (eg. by break)
        with contextlib.suppress(asyncio.CancelledError):
            await task
