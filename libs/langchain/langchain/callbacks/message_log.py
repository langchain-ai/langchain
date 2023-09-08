from __future__ import annotations
import math

from typing import Any, AsyncIterator, Callable, Optional, TypedDict, Union

import jsonpatch
from anyio import create_memory_object_stream

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


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

    def __init__(self, patch_log: list[dict], state: Optional[MessageState]) -> None:
        self.patch_log = patch_log
        self.state = state

    def __add__(self, other: Union[MessageLog, Any]) -> None:
        if isinstance(other, MessageLog):
            patch_log = self.patch_log + other.patch_log
            if self.state is None:
                return MessageLog(
                    patch_log,
                    jsonpatch.apply_patch(
                        MessageState(messages=[], final_output=None), patch_log
                    ),
                )
            else:
                return MessageLog(
                    patch_log, jsonpatch.apply_patch(self.state, other.patch_log)
                )

        return super().__add__(other)


class MessageLogCallbackHandler(BaseTracer):
    def __init__(self, run_filter: Callable[[Run], bool]) -> None:
        super().__init__()

        send_stream, receive_stream = create_memory_object_stream[MessageLog](math.inf)
        self.send_stream = send_stream
        self.receive_stream = receive_stream
        self.run_filter = run_filter

    def __aiter__(self) -> AsyncIterator[MessageLog]:
        return self.receive_stream.__aiter__()

    def _persist_run(self, run: Run) -> None:
        """The Langchain Tracer uses Post/Patch rather than persist."""

    def _persist_run_single(self, run: Run) -> None:
        """Start a run."""
        if run.parent_run_id is None:
            return

        if not self.run_filter(run):
            return

        self.send_stream.send_nowait(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": "/messages/-",
                        "value": Message(
                            metadata=run.extra.get("metadata", {}),
                            streamed_output=[],
                            final_output=None,
                        ),
                    }
                ],
            )
        )

    def _update_run_single(self, run: Run) -> None:
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
            self.send_stream.close()

        if not self.run_filter(run):
            return

        self.send_stream.send_nowait(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": "/messages/-/final_output",
                        "value": run.outputs,
                    }
                ],
            )
        )

    def _on_llm_new_token(self, run: Run, token: str) -> None:
        """Process new LLM token."""
        if not self.run_filter(run):
            return

        self.send_stream.send_nowait(
            MessageLog(
                patch_log=[
                    {
                        "op": "add",
                        "path": "/-/streamed_output/-",
                        "value": token,
                    }
                ],
            )
        )

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self._persist_run_single(run)

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self._persist_run_single(run)

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        self._update_run_single(run)

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self._update_run_single(run)

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        self._persist_run_single(run)

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        self._update_run_single(run)

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        self._update_run_single(run)

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        self._persist_run_single(run)

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self._update_run_single(run)

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self._update_run_single(run)

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""
        self._persist_run_single(run)

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""
        self._update_run_single(run)

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
        self._update_run_single(run)
