"""Stream transformer that emits only the final agent answer."""

from __future__ import annotations

from typing import Any, ClassVar

from langgraph.stream import StreamTransformer
from typing_extensions import override

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT


class _FinalAnswerStreamTransformer(StreamTransformer):
    """Suppress intermediate model chunks; emit only final answer chunks."""

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    @override
    def transform(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("event") != "on_chat_model_stream":
            return event

        data = event.get("data", {})
        chunk = data.get("chunk")
        chunk_position = getattr(chunk, "chunk_position", None)
        if chunk_position == "last":
            return event
        return None


class FinalAnswerMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that streams only the final model response chunks."""

    @property
    @override
    def transformers(self) -> list[StreamTransformer]:
        return [_FinalAnswerStreamTransformer()]
