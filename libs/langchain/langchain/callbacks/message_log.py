from __future__ import annotations

import asyncio
import math
from typing import Any, AsyncIterator, Callable, Dict, Optional, TypedDict, Union
from uuid import UUID

import jsonpatch
from anyio import create_memory_object_stream

from langchain.callbacks.tracers.base_async import AsyncBaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema.output import ChatGenerationChunk, GenerationChunk


class Message(TypedDict):
    name: str
    run_type: str
    tags: list[str]
    metadata: dict[str, Any]

    streamed_output: list[str]
    final_output: Optional[Any]


class MessageState(TypedDict):
    messages: list[Message]
    output: Optional[Any]


class MessageLog:
    state: Optional[MessageState]
    """List of messages."""
    patch_log: list[dict]
    """List of jsonpatch operations that created the messages."""

    def __init__(
        self, patch_log: list[dict], state: Optional[MessageState] = None
    ) -> None:
        self.patch_log = patch_log
        self.state = state

    def __add__(self, other: Union[MessageLog, Any]) -> MessageLog:
        if isinstance(other, MessageLog):
            patch_log = self.patch_log + other.patch_log
            if self.state is None:
                return MessageLog(
                    patch_log,
                    jsonpatch.apply_patch(
                        MessageState(messages=[], output=None), patch_log
                    ),
                )
            else:
                return MessageLog(
                    patch_log, jsonpatch.apply_patch(self.state, other.patch_log)
                )

        raise TypeError(
            f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        )

    def __repr__(self) -> str:
        from pprint import pformat

        if self.state is None:
            return f"MessageLog(patch_log={pformat(self.patch_log)})"
        else:
            return f"MessageLog(state={pformat(self.state)})"


class MessageLogCallbackHandler(AsyncBaseTracer):
    def __init__(self, run_filter: Callable[[Run], bool]) -> None:
        super().__init__()

        send_stream, receive_stream = create_memory_object_stream[MessageLog](math.inf)
        self.lock = asyncio.Lock()
        self.send_stream = send_stream
        self.receive_stream = receive_stream
        self.run_filter = run_filter
        self.index_map: Dict[UUID, int] = {}

    def __aiter__(self) -> AsyncIterator[MessageLog]:
        return self.receive_stream.__aiter__()

    async def _on_run_create(self, run: Run) -> None:
        """Start a run."""
        if run.parent_run_id is None:
            return

        if not self.run_filter(run):
            return

        # Determine previous index, increment by 1
        async with self.lock:
            previous_index = next(reversed(self.index_map.values()), -1)
            self.index_map[run.id] = previous_index + 1

        # Add the message to the stream
        await self.send_stream.send(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": f"/messages/{self.index_map[run.id]}",
                        "value": Message(
                            name=run.name,
                            run_type=run.run_type,
                            tags=run.tags or [],
                            metadata=run.extra.get("metadata", {}),
                            streamed_output=[],
                            final_output=None,
                        ),
                    }
                ],
            )
        )

    async def _on_run_update(self, run: Run) -> None:
        """Finish a run."""
        if run.parent_run_id is None:
            self.send_stream.send_nowait(
                MessageLog(
                    patch_log=[
                        {
                            "op": "replace",
                            "path": "/output",
                            "value": run.outputs,
                        }
                    ],
                )
            )
            return self.send_stream.close()

        if not self.run_filter(run):
            return

        index = self.index_map.get(run.id)

        if index is None:
            return

        await self.send_stream.send(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": f"/messages/{index}/final_output",
                        "value": run.outputs,
                    }
                ],
            )
        )

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""
        if not self.run_filter(run):
            return

        index = self.index_map.get(run.id)

        if index is None:
            return

        await self.send_stream.send(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": f"/messages/{index}/streamed_output/-",
                        "value": token,
                    }
                ],
            )
        )
