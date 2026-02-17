"""Tests for runnable event schema typing."""

from typing import get_args

from langchain_core.runnables.schema import StreamEventName


def test_stream_event_name_literal_values() -> None:
    """Test `StreamEventName` includes the supported `astream_events` names."""
    assert set(get_args(StreamEventName)) == {
        "on_llm_start",
        "on_llm_stream",
        "on_llm_end",
        "on_chat_model_start",
        "on_chat_model_stream",
        "on_chat_model_end",
        "on_tool_start",
        "on_tool_stream",
        "on_tool_end",
        "on_tool_error",
        "on_chain_start",
        "on_chain_stream",
        "on_chain_end",
        "on_retriever_start",
        "on_retriever_stream",
        "on_retriever_end",
        "on_prompt_start",
        "on_prompt_stream",
        "on_prompt_end",
        "on_parser_start",
        "on_parser_stream",
        "on_parser_end",
        "on_custom_event",
    }
