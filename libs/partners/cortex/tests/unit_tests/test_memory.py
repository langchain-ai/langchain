"""Unit tests for CortexMemory using a mocked CortexChatMessageHistory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_history(messages=None):
    """Return a MagicMock with the CortexChatMessageHistory interface."""
    mock = MagicMock()
    type(mock).messages = PropertyMock(return_value=messages or [])
    mock.get_context.return_value = "context"
    mock.cortex = MagicMock()
    return mock


def _make_memory(mock_history, **kwargs):
    """Instantiate CortexMemory, replacing the underlying chat history with mock."""
    from langchain_cortex.memory import CortexMemory

    mem = CortexMemory(db_path="/tmp/fake.db", session_id="s1", **kwargs)
    mem._chat_history = mock_history
    return mem


# ---------------------------------------------------------------------------
# Tests: memory_variables
# ---------------------------------------------------------------------------


def test_memory_variables_returns_memory_key() -> None:
    mem = _make_memory(_make_mock_history())
    assert mem.memory_variables == ["history"]


def test_memory_variables_respects_custom_memory_key() -> None:
    mem = _make_memory(_make_mock_history(), memory_key="chat_history")
    assert mem.memory_variables == ["chat_history"]


# ---------------------------------------------------------------------------
# Tests: load_memory_variables — return_messages=True
# ---------------------------------------------------------------------------


def test_load_memory_variables_returns_messages_list() -> None:
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    mem = _make_memory(_make_mock_history(msgs), return_messages=True)

    result = mem.load_memory_variables({})
    assert result == {"history": msgs}


def test_load_memory_variables_empty_history_returns_empty_list() -> None:
    mem = _make_memory(_make_mock_history([]), return_messages=True)
    result = mem.load_memory_variables({})
    assert result == {"history": []}


# ---------------------------------------------------------------------------
# Tests: load_memory_variables — return_messages=False (string mode)
# ---------------------------------------------------------------------------


def test_load_memory_variables_string_mode() -> None:
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    mem = _make_memory(_make_mock_history(msgs), return_messages=False)

    result = mem.load_memory_variables({})
    text = result["history"]
    assert "Human: hi" in text
    assert "AI: hello" in text


def test_load_memory_variables_string_mode_custom_prefixes() -> None:
    msgs = [HumanMessage(content="q"), AIMessage(content="a")]
    mem = _make_memory(
        _make_mock_history(msgs),
        return_messages=False,
        human_prefix="User",
        ai_prefix="Bot",
    )

    result = mem.load_memory_variables({})
    text = result["history"]
    assert text.startswith("User: q")
    assert "Bot: a" in text


# ---------------------------------------------------------------------------
# Tests: save_context
# ---------------------------------------------------------------------------


def test_save_context_adds_human_and_ai_messages() -> None:
    mock_history = _make_mock_history()
    mem = _make_memory(mock_history)

    mem.save_context({"input": "Hello"}, {"output": "Hi"})

    mock_history.add_messages.assert_called_once()
    added = mock_history.add_messages.call_args[0][0]
    assert len(added) == 2
    assert isinstance(added[0], HumanMessage)
    assert added[0].content == "Hello"
    assert isinstance(added[1], AIMessage)
    assert added[1].content == "Hi"


def test_save_context_uses_input_key_when_specified() -> None:
    mock_history = _make_mock_history()
    mem = _make_memory(mock_history, input_key="question", output_key="answer")

    mem.save_context(
        {"question": "What?", "metadata": "x"},
        {"answer": "42", "tokens": 10},
    )

    added = mock_history.add_messages.call_args[0][0]
    assert added[0].content == "What?"
    assert added[1].content == "42"


def test_save_context_falls_back_to_first_key() -> None:
    """When input_key/output_key are None, the first dict key is used."""
    mock_history = _make_mock_history()
    mem = _make_memory(mock_history)  # input_key=None, output_key=None

    mem.save_context({"question": "Hey"}, {"answer": "Yes"})

    added = mock_history.add_messages.call_args[0][0]
    assert added[0].content == "Hey"
    assert added[1].content == "Yes"


# ---------------------------------------------------------------------------
# Tests: clear
# ---------------------------------------------------------------------------


def test_clear_delegates_to_history() -> None:
    mock_history = _make_mock_history()
    mem = _make_memory(mock_history)
    mem.clear()
    mock_history.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: get_context
# ---------------------------------------------------------------------------


def test_get_context_delegates_to_history() -> None:
    mock_history = _make_mock_history()
    mock_history.get_context.return_value = "summary text"
    mem = _make_memory(mock_history)

    result = mem.get_context(max_tokens=1024)
    assert result == "summary text"
    mock_history.get_context.assert_called_once_with(1024)


# ---------------------------------------------------------------------------
# Tests: add_fact / add_preference
# ---------------------------------------------------------------------------


def test_add_fact_calls_cortex_add_fact() -> None:
    mock_history = _make_mock_history()
    mock_history.cortex.add_fact.return_value = "fact-id"
    mem = _make_memory(mock_history)

    result = mem.add_fact("User", "works_at", "Acme", 0.95)
    assert result == "fact-id"
    mock_history.cortex.add_fact.assert_called_once_with(
        "User", "works_at", "Acme", 0.95, "chat"
    )


def test_add_preference_calls_cortex_add_preference() -> None:
    mock_history = _make_mock_history()
    mock_history.cortex.add_preference.return_value = "pref-id"
    mem = _make_memory(mock_history)

    result = mem.add_preference("editor", "neovim", 0.9)
    assert result == "pref-id"
    mock_history.cortex.add_preference.assert_called_once_with("editor", "neovim", 0.9)


# ---------------------------------------------------------------------------
# Tests: cortex property
# ---------------------------------------------------------------------------


def test_cortex_property_exposes_underlying_instance() -> None:
    mock_history = _make_mock_history()
    fake_cx = MagicMock()
    mock_history.cortex = fake_cx
    mem = _make_memory(mock_history)

    assert mem.cortex is fake_cx
