"""Integration tests for CortexChatMessageHistory and CortexMemory.

These tests require ``cortex-ai-memory`` to be installed and run against a
real (temporary) Cortex database.  They are intentionally excluded from CI
unless the ``cortex-ai-memory`` package is present.

Run with::

    pytest tests/integration_tests/ -v -m requires
"""

from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytestmark = pytest.mark.requires("cortex_python")


@pytest.fixture()
def db_path(tmp_path):
    """Provide a temp directory Cortex db path for each test."""
    return str(tmp_path / "test_cortex.db")


# ---------------------------------------------------------------------------
# CortexChatMessageHistory
# ---------------------------------------------------------------------------

class TestCortexChatMessageHistoryIntegration:
    def test_add_and_retrieve_messages(self, db_path) -> None:
        from langchain_cortex import CortexChatMessageHistory

        history = CortexChatMessageHistory(db_path=db_path, session_id="s1")
        history.add_messages([
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ])

        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there!"

    def test_multiple_sessions_are_isolated(self, db_path) -> None:
        from langchain_cortex import CortexChatMessageHistory

        h1 = CortexChatMessageHistory(db_path=db_path, session_id="session-A")
        h2 = CortexChatMessageHistory(db_path=db_path, session_id="session-B")

        h1.add_messages([HumanMessage(content="from A")])
        h2.add_messages([HumanMessage(content="from B")])

        msgs_a = h1.messages
        msgs_b = h2.messages

        assert all("from A" in m.content for m in msgs_a)
        assert all("from B" in m.content for m in msgs_b)

    def test_clear_marks_session_cleared(self, db_path) -> None:
        from langchain_cortex import CortexChatMessageHistory

        history = CortexChatMessageHistory(db_path=db_path, session_id="clearme")
        history.add_messages([HumanMessage(content="before clear")])
        history.clear()
        # After clear(), new messages added should still be retrievable.
        history.add_messages([HumanMessage(content="after clear")])
        msgs = history.messages
        # At minimum the post-clear message should be present.
        assert any("after clear" in m.content for m in msgs)

    def test_get_context_returns_string(self, db_path) -> None:
        from langchain_cortex import CortexChatMessageHistory

        history = CortexChatMessageHistory(db_path=db_path, session_id="ctx-test")
        history.add_messages([HumanMessage(content="test context")])
        ctx = history.get_context(max_tokens=500)
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# CortexMemory
# ---------------------------------------------------------------------------

class TestCortexMemoryIntegration:
    def test_save_and_load(self, db_path) -> None:
        from langchain_cortex import CortexMemory

        mem = CortexMemory(db_path=db_path, session_id="mem-test")
        mem.save_context({"input": "Who are you?"}, {"output": "I am an AI."})

        result = mem.load_memory_variables({})
        msgs = result["history"]
        assert len(msgs) == 2
        assert msgs[0].content == "Who are you?"
        assert msgs[1].content == "I am an AI."

    def test_clear_resets_history(self, db_path) -> None:
        from langchain_cortex import CortexMemory

        mem = CortexMemory(db_path=db_path, session_id="clear-test")
        mem.save_context({"input": "Hello"}, {"output": "Hi"})
        mem.clear()
        mem.save_context({"input": "After clear"}, {"output": "Yes"})

        result = mem.load_memory_variables({})
        msgs = result["history"]
        assert any("After clear" in m.content for m in msgs)

    def test_add_fact_stores_in_cortex(self, db_path) -> None:
        from langchain_cortex import CortexMemory

        mem = CortexMemory(db_path=db_path, session_id="facts-test")
        fact_id = mem.add_fact("User", "lives_in", "Tokyo", 0.9)
        assert isinstance(fact_id, str)
        assert len(fact_id) > 0

    def test_add_preference_stores_in_cortex(self, db_path) -> None:
        from langchain_cortex import CortexMemory

        mem = CortexMemory(db_path=db_path, session_id="pref-test")
        pref_id = mem.add_preference("theme", "dark", 0.95)
        assert isinstance(pref_id, str)
        assert len(pref_id) > 0
