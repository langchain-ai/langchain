"""Unit tests for prompt_cache_key parameter."""

from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI


def test_prompt_cache_key_parameter_inclusion() -> None:
    """Test that prompt_cache_key parameter is properly included in request payload."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(messages, prompt_cache_key="test-cache-key")
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "test-cache-key"


def test_prompt_cache_key_parameter_exclusion() -> None:
    """Test that prompt_cache_key parameter behavior matches OpenAI API."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    # Test with explicit None (OpenAI should accept None values (marked Optional))
    payload = chat._get_request_payload(messages, prompt_cache_key=None)
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] is None
