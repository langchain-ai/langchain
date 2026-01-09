"""Tests for runnables schema types."""

from typing import get_args

import pytest

from langchain_core.runnables import StreamEventName
from langchain_core.runnables.schema import (
    StreamEventName as SchemaStreamEventName,
)
from langchain_core.runnables.schema import (
    get_event_name,
)

# Expected set of all valid event names
# Update this set when adding/removing events from StreamEventName
EXPECTED_EVENT_NAMES = frozenset(
    {
        # LLM (non-chat models)
        "on_llm_start",
        "on_llm_stream",
        "on_llm_end",
        "on_llm_error",
        # Chat Model
        "on_chat_model_start",
        "on_chat_model_stream",
        "on_chat_model_end",
        # Tool
        "on_tool_start",
        "on_tool_stream",
        "on_tool_end",
        "on_tool_error",
        # Chain
        "on_chain_start",
        "on_chain_stream",
        "on_chain_end",
        "on_chain_error",
        # Retriever
        "on_retriever_start",
        "on_retriever_stream",
        "on_retriever_end",
        "on_retriever_error",
        # Prompt
        "on_prompt_start",
        "on_prompt_end",
        # Parser
        "on_parser_start",
        "on_parser_stream",
        "on_parser_end",
        "on_parser_error",
        # Custom
        "on_custom_event",
    }
)


def test_stream_event_name_exported() -> None:
    """Test that StreamEventName is properly exported from runnables module."""
    assert StreamEventName is SchemaStreamEventName


def test_stream_event_name_matches_expected() -> None:
    """Test that StreamEventName contains exactly the expected event names."""
    actual_events = frozenset(get_args(StreamEventName))
    assert actual_events == EXPECTED_EVENT_NAMES, (
        f"StreamEventName mismatch.\n"
        f"Missing: {EXPECTED_EVENT_NAMES - actual_events}\n"
        f"Extra: {actual_events - EXPECTED_EVENT_NAMES}"
    )


def test_stream_event_name_is_literal_str_subtype() -> None:
    """Test that StreamEventName values are strings (backward compatible)."""
    event_names = get_args(StreamEventName)
    for name in event_names:
        assert isinstance(name, str)
        assert name.startswith("on_")


def test_get_event_name_valid_combinations() -> None:
    """Test that get_event_name returns correct event names."""
    # Test all run_type/phase combinations
    assert get_event_name("llm", "start") == "on_llm_start"
    assert get_event_name("llm", "stream") == "on_llm_stream"
    assert get_event_name("llm", "end") == "on_llm_end"
    assert get_event_name("llm", "error") == "on_llm_error"

    assert get_event_name("chat_model", "start") == "on_chat_model_start"
    assert get_event_name("chat_model", "stream") == "on_chat_model_stream"
    assert get_event_name("chat_model", "end") == "on_chat_model_end"

    assert get_event_name("tool", "start") == "on_tool_start"
    assert get_event_name("tool", "stream") == "on_tool_stream"
    assert get_event_name("tool", "end") == "on_tool_end"
    assert get_event_name("tool", "error") == "on_tool_error"

    assert get_event_name("chain", "start") == "on_chain_start"
    assert get_event_name("chain", "stream") == "on_chain_stream"
    assert get_event_name("chain", "end") == "on_chain_end"
    assert get_event_name("chain", "error") == "on_chain_error"

    assert get_event_name("retriever", "start") == "on_retriever_start"
    assert get_event_name("retriever", "stream") == "on_retriever_stream"
    assert get_event_name("retriever", "end") == "on_retriever_end"
    assert get_event_name("retriever", "error") == "on_retriever_error"

    assert get_event_name("prompt", "start") == "on_prompt_start"
    assert get_event_name("prompt", "end") == "on_prompt_end"

    assert get_event_name("parser", "start") == "on_parser_start"
    assert get_event_name("parser", "stream") == "on_parser_stream"
    assert get_event_name("parser", "end") == "on_parser_end"
    assert get_event_name("parser", "error") == "on_parser_error"


def test_get_event_name_invalid_run_type() -> None:
    """Test that get_event_name raises ValueError for invalid run_type."""
    with pytest.raises(ValueError, match="Unknown run_type"):
        get_event_name("invalid_type", "start")


def test_get_event_name_invalid_phase() -> None:
    """Test that get_event_name raises ValueError for invalid phase."""
    # chat_model doesn't have an "error" phase
    with pytest.raises(ValueError, match=r"Phase .* is not valid"):
        get_event_name("chat_model", "error")

    # prompt doesn't have "stream" or "error" phases
    with pytest.raises(ValueError, match=r"Phase .* is not valid"):
        get_event_name("prompt", "stream")
