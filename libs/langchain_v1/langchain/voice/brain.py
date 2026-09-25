"""Adapter from background task instructions to a LangGraph-like graph."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Protocol

MAX_RESULT_CHARS = 48_000


class AsyncGraph(Protocol):
    """The part of a compiled LangGraph that LangChain Voice uses."""

    def ainvoke(
        self, input_value: Any, config: dict[str, Any] | None = None
    ) -> Any:  # pragma: no cover - structural typing declaration
        """Invoke the graph with one task input and optional configuration."""
        ...


InputFactory = Callable[[str], Any]
ResultFormatter = Callable[[Any], str]


def default_input_factory(instruction: str) -> dict[str, list[dict[str, str]]]:
    """Wrap an instruction in the default LangGraph messages state."""
    return {"messages": [{"role": "user", "content": instruction}]}


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for part in value:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _message_text(message: Any) -> str:
    if isinstance(message, Mapping):
        return _content_text(message.get("content"))
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return _content_text(getattr(message, "content", None))


def default_result_formatter(result: Any) -> str:
    """Extract a useful final string from common LangGraph result shapes."""
    if isinstance(result, str):
        text = result
    elif isinstance(result, Mapping):
        text = ""
        for key in ("response", "output", "result"):
            if isinstance(result.get(key), str):
                text = result[key]
                break
        if not text and isinstance(result.get("messages"), list):
            for message in reversed(result["messages"]):
                text = _message_text(message)
                if text:
                    break
    else:
        text = _message_text(result)

    text = text.strip()
    if not text:
        msg = "The graph returned no text result"
        raise ValueError(msg)
    if len(text) > MAX_RESULT_CHARS:
        return text[: MAX_RESULT_CHARS - 14] + " …[truncated]"
    return text


class GraphBrain:
    """Run one natural-language instruction on one persistent graph thread."""

    def __init__(
        self,
        graph: AsyncGraph,
        *,
        input_factory: InputFactory | None = None,
        result_formatter: ResultFormatter | None = None,
    ) -> None:
        """Initialize the graph adapter and result conversion hooks."""
        if not callable(getattr(graph, "ainvoke", None)):
            msg = "graph must provide an ainvoke(input, config=...) method"
            raise TypeError(msg)
        self._graph = graph
        self._input_factory = input_factory or default_input_factory
        self._result_formatter = result_formatter or default_result_formatter

    async def run(self, instruction: str, *, thread_id: str) -> str:
        """Run an instruction on a persistent graph thread."""
        invocation = self._graph.ainvoke(
            self._input_factory(instruction),
            config={"configurable": {"thread_id": thread_id}},
        )
        result = await invocation if inspect.isawaitable(invocation) else invocation
        return self._result_formatter(result)
