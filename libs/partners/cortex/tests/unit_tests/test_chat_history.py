"""Unit tests for CortexChatMessageHistory using a mocked Cortex backend."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "test-session-42"
TAG = f"__session__{SESSION}::"


def _make_mock_cortex(stored: list[tuple[str, float, str]] | None = None) -> MagicMock:
    """Return a MagicMock that behaves like ``PyCortex``."""
    mock = MagicMock()
    mock.retrieve.return_value = stored or []
    mock.ingest.return_value = "fake-mem-id"
    return mock


def _make_history(mock_cx: MagicMock) -> Any:
    """Instantiate CortexChatMessageHistory with a pre-wired mock backend.

    We patch ``langchain_cortex.chat_history._open_cortex`` — the single
    factory that calls ``PyCortex(...)`` — so that no real database is opened.
    """
    from langchain_cortex.chat_history import CortexChatMessageHistory

    with patch(
        "langchain_cortex.chat_history._open_cortex",
        return_value=mock_cx,
    ):
        history = CortexChatMessageHistory(
            db_path="/tmp/fake.db",
            session_id=SESSION,
        )
    return history


# ---------------------------------------------------------------------------
# Tests: messages property
# ---------------------------------------------------------------------------


def test_messages_empty_when_no_stored_data() -> None:
    """messages returns [] when Cortex retrieve returns nothing."""
    history = _make_history(_make_mock_cortex([]))
    assert history.messages == []


def test_messages_reconstructed_in_order() -> None:
    """messages returns turns in ascending sequence order."""
    import json
    from langchain_core.messages import messages_to_dict

    human = HumanMessage(content="hello")
    ai = AIMessage(content="world")

    def pack(msg: Any, seq: int) -> str:
        payload = json.dumps(messages_to_dict([msg])[0])
        return f"{TAG}{seq}:{payload}"

    stored = [
        ("id-1", 0.9, pack(human, 0)),
        ("id-2", 0.85, pack(ai, 1)),
    ]
    history = _make_history(_make_mock_cortex(stored))

    msgs = history.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hello"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "world"


def test_messages_sorted_by_sequence_not_retrieval_order() -> None:
    """messages sorts by seq even when Cortex returns them out of order."""
    import json
    from langchain_core.messages import messages_to_dict

    turns = [HumanMessage(content=f"turn {i}") for i in range(4)]

    def pack(msg: Any, seq: int) -> str:
        payload = json.dumps(messages_to_dict([msg])[0])
        return f"{TAG}{seq}:{payload}"

    # Deliberately reverse retrieval order.
    stored = [("id", 0.9, pack(turns[i], i)) for i in reversed(range(4))]
    history = _make_history(_make_mock_cortex(stored))

    msgs = history.messages
    assert [m.content for m in msgs] == [f"turn {i}" for i in range(4)]


def test_messages_ignores_entries_from_other_sessions() -> None:
    """Entries tagged with a different session_id are silently skipped."""
    import json
    from langchain_core.messages import messages_to_dict

    human = HumanMessage(content="mine")
    other_tag = "__session__other-session::"
    my_tag = TAG

    def pack(tag: str, msg: Any, seq: int) -> str:
        payload = json.dumps(messages_to_dict([msg])[0])
        return f"{tag}{seq}:{payload}"

    stored = [
        ("id-1", 0.9, pack(other_tag, HumanMessage(content="not mine"), 0)),
        ("id-2", 0.8, pack(my_tag, human, 0)),
    ]
    history = _make_history(_make_mock_cortex(stored))
    msgs = history.messages
    assert len(msgs) == 1
    assert msgs[0].content == "mine"


# ---------------------------------------------------------------------------
# Tests: add_messages
# ---------------------------------------------------------------------------


def test_add_messages_calls_ingest_for_each_message() -> None:
    """add_messages must call cortex.ingest once per message."""
    mock_cx = _make_mock_cortex([])
    history = _make_history(mock_cx)

    messages = [HumanMessage(content="hi"), AIMessage(content="hey")]
    history.add_messages(messages)

    assert mock_cx.ingest.call_count == len(messages)


def test_add_messages_includes_session_tag_in_stored_text() -> None:
    """Each ingested text must start with the session tag."""
    mock_cx = _make_mock_cortex([])
    history = _make_history(mock_cx)

    history.add_messages([HumanMessage(content="test")])

    call_args = mock_cx.ingest.call_args_list[0]
    stored_text: str = call_args[0][0]
    assert stored_text.startswith(TAG)


def test_add_messages_sequence_numbers_are_contiguous() -> None:
    """Sequence numbers in stored texts start at current length and increment."""
    mock_cx = _make_mock_cortex([])
    history = _make_history(mock_cx)

    msgs = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
    history.add_messages(msgs)

    seqs = []
    for call in mock_cx.ingest.call_args_list:
        text: str = call[0][0]
        rest = text[len(TAG):]
        seq = int(rest.split(":")[0])
        seqs.append(seq)

    assert seqs == [0, 1, 2]


# ---------------------------------------------------------------------------
# Tests: clear
# ---------------------------------------------------------------------------


def test_clear_ingests_sentinel() -> None:
    """clear() must store a sentinel entry."""
    mock_cx = _make_mock_cortex([])
    history = _make_history(mock_cx)

    history.clear()

    assert mock_cx.ingest.call_count == 1
    sentinel_text: str = mock_cx.ingest.call_args[0][0]
    assert "__cleared__" in sentinel_text


# ---------------------------------------------------------------------------
# Tests: get_context
# ---------------------------------------------------------------------------


def test_get_context_delegates_to_cortex() -> None:
    """get_context() delegates to cortex.get_context with correct args."""
    mock_cx = _make_mock_cortex([])
    mock_cx.get_context.return_value = "context summary"
    history = _make_history(mock_cx)

    result = history.get_context(max_tokens=500)

    assert result == "context summary"
    mock_cx.get_context.assert_called_once_with(500, channel="chat")


# ---------------------------------------------------------------------------
# Tests: import error
# ---------------------------------------------------------------------------


def test_import_error_raised_without_cortex_package() -> None:
    """A helpful ImportError is raised when cortex-ai-memory is not installed."""
    # Simulate absence of cortex_python by patching _open_cortex to raise.
    with patch(
        "langchain_cortex.chat_history._open_cortex",
        side_effect=ImportError("Could not import cortex-ai-memory. Install it with: pip install cortex-ai-memory"),
    ):
        from langchain_cortex.chat_history import CortexChatMessageHistory

        with pytest.raises(ImportError, match="cortex-ai-memory"):
            CortexChatMessageHistory(db_path="/tmp/fake.db", session_id="s1")
