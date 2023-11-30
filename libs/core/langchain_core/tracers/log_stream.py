from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
)
from uuid import UUID

import jsonpatch  # type: ignore[import]
from anyio import create_memory_object_stream

from langchain_core.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class LogEntry(TypedDict):
    """A single entry in the run log."""

    id: str
    """ID of the sub-run."""
    name: str
    """Name of the object being run."""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc."""
    tags: List[str]
    """List of tags for the run."""
    metadata: Dict[str, Any]
    """Key-value pairs of metadata for the run."""
    start_time: str
    """ISO-8601 timestamp of when the run started."""

    streamed_output_str: List[str]
    """List of LLM tokens streamed by this run, if applicable."""
    final_output: Optional[Any]
    """Final output of this run.
    Only available after the run has finished successfully."""
    end_time: Optional[str]
    """ISO-8601 timestamp of when the run ended.
    Only available after the run has finished."""


class RunState(TypedDict):
    """State of the run."""

    id: str
    """ID of the run."""
    streamed_output: List[Any]
    """List of output chunks streamed by Runnable.stream()"""
    final_output: Optional[Any]
    """Final output of the run, usually the result of aggregating (`+`) streamed_output.
    Updated throughout the run when supported by the Runnable."""

    logs: Dict[str, LogEntry]
    """Map of run names to sub-runs. If filters were supplied, this list will
    contain only the runs that matched the filters."""


class RunLogPatch:
    """A patch to the run log."""

    ops: List[Dict[str, Any]]
    """List of jsonpatch operations, which describe how to create the run state
    from an empty dict. This is the minimal representation of the log, designed to
    be serialized as JSON and sent over the wire to reconstruct the log on the other
    side. Reconstruction of the state can be done with any jsonpatch-compliant library,
    see https://jsonpatch.com for more information."""

    def __init__(self, *ops: Dict[str, Any]) -> None:
        self.ops = list(ops)

    def __add__(self, other: Union[RunLogPatch, Any]) -> RunLog:
        if type(other) == RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(None, ops)
            return RunLog(*ops, state=state)

        raise TypeError(
            f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        )

    def __repr__(self) -> str:
        from pprint import pformat

        # 1:-1 to get rid of the [] around the list
        return f"RunLogPatch({pformat(self.ops)[1:-1]})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunLogPatch) and self.ops == other.ops


class RunLog(RunLogPatch):
    """A run log."""

    state: RunState
    """Current state of the log, obtained from applying all ops in sequence."""

    def __init__(self, *ops: Dict[str, Any], state: RunState) -> None:
        super().__init__(*ops)
        self.state = state

    def __add__(self, other: Union[RunLogPatch, Any]) -> RunLog:
        if type(other) == RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(self.state, other.ops)
            return RunLog(*ops, state=state)

        raise TypeError(
            f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        )

    def __repr__(self) -> str:
        from pprint import pformat

        return f"RunLog({pformat(self.state)})"


class LogStreamCallbackHandler(BaseTracer):
    """A tracer that streams run logs to a stream."""

    def __init__(
        self,
        *,
        auto_close: bool = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        self.auto_close = auto_close
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

        send_stream: Any
        receive_stream: Any
        send_stream, receive_stream = create_memory_object_stream(math.inf)
        self.lock = threading.Lock()
        self.send_stream = send_stream
        self.receive_stream = receive_stream
        self._key_map_by_run_id: Dict[UUID, str] = {}
        self._counter_map_by_name: Dict[str, int] = defaultdict(int)
        self.root_id: Optional[UUID] = None

    def __aiter__(self) -> AsyncIterator[RunLogPatch]:
        return self.receive_stream.__aiter__()

    def include_run(self, run: Run) -> bool:
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
            self.send_stream.send_nowait(
                RunLogPatch(
                    {
                        "op": "replace",
                        "path": "",
                        "value": RunState(
                            id=str(run.id),
                            streamed_output=[],
                            final_output=None,
                            logs={},
                        ),
                    }
                )
            )

        if not self.include_run(run):
            return

        # Determine previous index, increment by 1
        with self.lock:
            self._counter_map_by_name[run.name] += 1
            count = self._counter_map_by_name[run.name]
            self._key_map_by_run_id[run.id] = (
                run.name if count == 1 else f"{run.name}:{count}"
            )

        # Add the run to the stream
        self.send_stream.send_nowait(
            RunLogPatch(
                {
                    "op": "add",
                    "path": f"/logs/{self._key_map_by_run_id[run.id]}",
                    "value": LogEntry(
                        id=str(run.id),
                        name=run.name,
                        type=run.run_type,
                        tags=run.tags or [],
                        metadata=(run.extra or {}).get("metadata", {}),
                        start_time=run.start_time.isoformat(timespec="milliseconds"),
                        streamed_output_str=[],
                        final_output=None,
                        end_time=None,
                    ),
                }
            )
        )

    def _on_run_update(self, run: Run) -> None:
        """Finish a run."""
        try:
            index = self._key_map_by_run_id.get(run.id)

            if index is None:
                return

            self.send_stream.send_nowait(
                RunLogPatch(
                    {
                        "op": "add",
                        "path": f"/logs/{index}/final_output",
                        # to undo the dumpd done by some runnables / tracer / etc
                        "value": load(run.outputs),
                    },
                    {
                        "op": "add",
                        "path": f"/logs/{index}/end_time",
                        "value": run.end_time.isoformat(timespec="milliseconds")
                        if run.end_time is not None
                        else None,
                    },
                )
            )
        finally:
            if run.id == self.root_id:
                if self.auto_close:
                    self.send_stream.close()

    def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""
        index = self._key_map_by_run_id.get(run.id)

        if index is None:
            return

        self.send_stream.send_nowait(
            RunLogPatch(
                {
                    "op": "add",
                    "path": f"/logs/{index}/streamed_output_str/-",
                    "value": token,
                }
            )
        )
