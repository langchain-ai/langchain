from __future__ import annotations

import unittest
from typing import Any

import pytest

from langchain.voice.brain import GraphBrain, default_result_formatter


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((state, config))
        return {"messages": [*state["messages"], {"content": "done"}]}


class GraphBrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_invokes_graph_on_requested_thread(self) -> None:
        graph = RecordingGraph()
        brain = GraphBrain(graph)

        result = await brain.run("search Heathrow", thread_id="thread-1")

        assert result == "done"
        assert graph.calls[0][0]["messages"][0]["content"] == "search Heathrow"
        assert graph.calls[0][1] == {"configurable": {"thread_id": "thread-1"}}

    def test_formats_common_results(self) -> None:
        assert default_result_formatter({"response": "hello"}) == "hello"
        result = {"messages": [{"content": "first"}, {"content": "last"}]}
        assert default_result_formatter(result) == "last"

    def test_uses_message_text_property(self) -> None:
        class PropertyMessage:
            text = "from property"
            content = "from content"

        assert default_result_formatter({"messages": [PropertyMessage()]}) == "from property"

    def test_does_not_call_deprecated_message_text_method(self) -> None:
        class LegacyMessage:
            content = "from content"

            def text(self) -> str:
                msg = "deprecated text() method was called"
                raise AssertionError(msg)

        assert default_result_formatter({"messages": [LegacyMessage()]}) == "from content"

    def test_rejects_empty_result(self) -> None:
        with pytest.raises(ValueError, match="no text"):
            default_result_formatter({"messages": []})


if __name__ == "__main__":
    unittest.main()
