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


def test_prompt_cache_key_per_call() -> None:
    """Test that prompt_cache_key can be passed per-call with different values."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    # Test different cache keys per call
    payload1 = chat._get_request_payload(messages, prompt_cache_key="cache-v1")
    payload2 = chat._get_request_payload(messages, prompt_cache_key="cache-v2")

    assert payload1["prompt_cache_key"] == "cache-v1"
    assert payload2["prompt_cache_key"] == "cache-v2"

    # Test dynamic cache key assignment
    cache_keys = ["customer-v1", "support-v1", "feedback-v1"]

    for cache_key in cache_keys:
        payload = chat._get_request_payload(messages, prompt_cache_key=cache_key)
        assert "prompt_cache_key" in payload
        assert payload["prompt_cache_key"] == cache_key


def test_prompt_cache_key_model_kwargs() -> None:
    """Test prompt_cache_key via model_kwargs and method precedence."""
    messages = [HumanMessage("Hello world")]

    # Test model-level via model_kwargs
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        max_completion_tokens=10,
        model_kwargs={"prompt_cache_key": "model-level-cache"},
    )
    payload = chat._get_request_payload(messages)
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "model-level-cache"

    # Test that per-call cache key overrides model-level
    payload_override = chat._get_request_payload(
        messages, prompt_cache_key="per-call-cache"
    )
    assert payload_override["prompt_cache_key"] == "per-call-cache"


def test_prompt_cache_key_responses_api() -> None:
    """Test that prompt_cache_key works with Responses API."""
    chat = ChatOpenAI(
        model="gpt-4o-mini", use_responses_api=True, max_completion_tokens=10
    )

    messages = [HumanMessage("Hello")]
    payload = chat._get_request_payload(
        messages, prompt_cache_key="responses-api-cache-v1"
    )

    # prompt_cache_key should be present regardless of API type
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "responses-api-cache-v1"
