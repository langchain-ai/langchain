from __future__ import annotations

import asyncio
import math
from ast import List
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Sequence,
    TypedDict,
    Union,
)
from uuid import UUID

import jsonpatch
from anyio import create_memory_object_stream

from langchain.callbacks.tracers.base_async import AsyncBaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema.output import ChatGenerationChunk, GenerationChunk


class LogEntry(TypedDict):
    id: str
    name: str
    type: str
    tags: List[str]
    metadata: Dict[str, Any]
    start_time: str

    streamed_output_str: List[str]
    final_output: Optional[Any]
    end_time: Optional[str]


class LogState(TypedDict):
    logs: list[LogEntry]
    streamed_output: List[Any]
    final_output: Optional[Any]


class Log:
    state: Optional[LogState]
    """Current state of the log, obtained from applying all ops to an empty dict."""
    ops: List[Dict[str, Any]]
    """List of jsonpatch operations that created state."""

    def __init__(
        self, ops: List[Dict[str, Any]], *, state: Optional[LogState] = None
    ) -> None:
        self.ops = ops
        self.state = state

    def __add__(self, other: Union[Log, Any]) -> Log:
        if isinstance(other, Log):
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(
                self.state, ops if self.state is None else other.ops
            )
            return Log(ops, state=state)

        raise TypeError(
            f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        )

    def __repr__(self) -> str:
        from pprint import pformat

        if self.state is None:
            return f"Log(ops={pformat(self.ops)})"
        else:
            return f"Log(state={pformat(self.state)})"


class LogStreamCallbackHandler(AsyncBaseTracer):
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

        send_stream, receive_stream = create_memory_object_stream[Log](math.inf)
        self.lock = asyncio.Lock()
        self.send_stream = send_stream
        self.receive_stream = receive_stream
        self._index_map: Dict[UUID, int] = {}

    def __aiter__(self) -> AsyncIterator[Log]:
        return self.receive_stream.__aiter__()

    def include_run(self, run: Run) -> bool:
        if run.parent_run_id is None:
            include = False
            return include

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

    async def _on_run_create(self, run: Run) -> None:
        """Start a run."""
        if run.parent_run_id is None:
            await self.send_stream.send(
                Log(
                    [
                        {
                            "op": "replace",
                            "path": "",
                            "value": LogState(
                                logs=[], streamed_output=[], final_output=None
                            ),
                        }
                    ]
                )
            )

        if not self.include_run(run):
            return

        # Determine previous index, increment by 1
        async with self.lock:
            previous_index = next(reversed(self._index_map.values()), -1)
            self._index_map[run.id] = previous_index + 1

        # Add the run to the stream
        await self.send_stream.send(
            Log(
                [
                    {
                        "op": "add",
                        "path": f"/logs/{self._index_map[run.id]}",
                        "value": LogEntry(
                            id=str(run.id),
                            name=run.name,
                            type=run.run_type,
                            tags=run.tags or [],
                            metadata=run.extra.get("metadata", {}),
                            start_time=run.start_time.isoformat(
                                timespec="milliseconds"
                            ),
                            streamed_output_str=[],
                            final_output=None,
                        ),
                    }
                ]
            )
        )

    async def _on_run_update(self, run: Run) -> None:
        """Finish a run."""
        try:
            index = self._index_map.get(run.id)

            if index is None:
                return

            await self.send_stream.send(
                Log(
                    [
                        {
                            "op": "add",
                            "path": f"/logs/{index}/final_output",
                            "value": run.outputs,
                        },
                        {
                            "op": "add",
                            "path": f"/logs/{index}/end_time",
                            "value": run.end_time.isoformat(timespec="milliseconds"),
                        },
                    ]
                )
            )
        finally:
            if run.parent_run_id is None:
                await self.send_stream.send(
                    Log(
                        [
                            {
                                "op": "replace",
                                "path": "/final_output",
                                "value": run.outputs,
                            }
                        ]
                    )
                )
                if self.auto_close:
                    await self.send_stream.aclose()

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""
        index = self._index_map.get(run.id)

        if index is None:
            return

        await self.send_stream.send(
            Log(
                [
                    {
                        "op": "add",
                        "path": f"/logs/{index}/streamed_output_str/-",
                        "value": token,
                    }
                ]
            )
        )
