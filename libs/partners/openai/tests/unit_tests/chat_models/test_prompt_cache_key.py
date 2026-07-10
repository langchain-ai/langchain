"""Unit tests for prompt_cache_key parameter."""

from langchain_core.messages import HumanMessage, ToolMessage

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
        model="gpt-4o-mini",
        use_responses_api=True,
        output_version="responses/v1",
        max_completion_tokens=10,
    )

    messages = [HumanMessage("Hello")]
    payload = chat._get_request_payload(
        messages, prompt_cache_key="responses-api-cache-v1"
    )

    # prompt_cache_key should be present regardless of API type
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "responses-api-cache-v1"


def test_prompt_cache_options_and_retention_request_payload() -> None:
    """Per-invocation cache options/retention flow into the request payload."""
    chat = ChatOpenAI(model="gpt-5.5", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(
        messages,
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        prompt_cache_retention="24h",
    )

    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["prompt_cache_retention"] == "24h"


def test_prompt_cache_options_and_retention_responses_api_payload() -> None:
    """Cache options/retention survive Responses API payload construction."""
    chat = ChatOpenAI(
        model="gpt-5.5",
        use_responses_api=True,
        output_version="responses/v1",
        max_completion_tokens=10,
    )
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(
        messages,
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        prompt_cache_retention="24h",
    )

    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["prompt_cache_retention"] == "24h"


def test_prompt_cache_options_and_retention_model_kwargs() -> None:
    """Cache options/retention set via `model_kwargs` flow into the payload."""
    chat = ChatOpenAI(
        model="gpt-5.5",
        max_completion_tokens=10,
        model_kwargs={
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_retention": "24h",
        },
    )
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(messages)

    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["prompt_cache_retention"] == "24h"


def test_prompt_cache_breakpoint_chat_completions_text_block() -> None:
    """A `prompt_cache_breakpoint` on a text block is preserved for Chat Completions."""
    chat = ChatOpenAI(model="gpt-5.5", max_completion_tokens=10)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Stable prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
        )
    ]

    payload = chat._get_request_payload(messages)

    assert payload["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


def test_prompt_cache_breakpoint_chat_completions_tool_message() -> None:
    """A cache breakpoint on a tool result text block is preserved."""
    chat = ChatOpenAI(model="gpt-5.5", max_completion_tokens=10)
    messages = [
        ToolMessage(
            tool_call_id="call_123",
            content=[
                {
                    "type": "text",
                    "text": "Stable tool result",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        )
    ]

    payload = chat._get_request_payload(messages)

    assert payload["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Stable tool result",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]


def test_prompt_cache_breakpoint_responses_api_converted_blocks() -> None:
    """`prompt_cache_breakpoint` survives Responses API conversion for each block."""
    chat = ChatOpenAI(
        model="gpt-5.5",
        use_responses_api=True,
        output_version="responses/v1",
        max_completion_tokens=10,
    )
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Stable text prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "file",
                    "file": {"file_id": "file_123"},
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
            ]
        )
    ]

    payload = chat._get_request_payload(messages)
    content = payload["input"][0]["content"]

    assert content == [
        {
            "type": "input_text",
            "text": "Stable text prefix",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "input_image",
            "image_url": "https://example.com/image.png",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "input_file",
            "file_id": "file_123",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
    ]
