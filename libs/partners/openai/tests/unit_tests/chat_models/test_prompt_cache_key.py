"""Unit tests for prompt_cache_key parameter."""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.content import create_text_block

from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-5.5"


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
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
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
        model=MODEL_NAME,
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


def test_prompt_cache_options_init_param() -> None:
    """Model-level cache options flow into the payload and can be overridden."""
    chat = ChatOpenAI(
        model=MODEL_NAME,
        max_completion_tokens=10,
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        model_kwargs={"prompt_cache_retention": "24h"},
    )
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(messages)
    override_payload = chat._get_request_payload(
        messages, prompt_cache_options={"mode": "implicit"}
    )

    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["prompt_cache_retention"] == "24h"
    assert override_payload["prompt_cache_options"] == {"mode": "implicit"}


def test_prompt_cache_breakpoint_chat_completions_text_block() -> None:
    """A `prompt_cache_breakpoint` on a text block is preserved for Chat Completions."""
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
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
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
    messages = [
        ToolMessage(
            tool_call_id="call_123",
            content_blocks=[
                {
                    "type": "text",
                    "text": "Stable tool result",
                    "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
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
        model=MODEL_NAME,
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


def test_prompt_cache_breakpoint_chat_completions_content_blocks() -> None:
    """Standard content block cache breakpoints reach Chat Completions."""
    chat = ChatOpenAI(model=MODEL_NAME, output_version="v1", max_completion_tokens=10)
    message = HumanMessage(
        content_blocks=[
            create_text_block(
                "Stable text prefix",
                prompt_cache_breakpoint={"mode": "explicit"},
            ),
            {
                "type": "image",
                "url": "https://example.com/image.png",
                "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
            },
            {
                "type": "file",
                "file_id": "file_123",
                "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
            },
        ]
    )

    payload = chat._get_request_payload([message])

    assert payload["messages"][0]["content"] == [
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


def test_prompt_cache_breakpoint_responses_api_content_blocks() -> None:
    """Standard content block cache breakpoints reach the Responses API."""
    chat = ChatOpenAI(
        model=MODEL_NAME,
        use_responses_api=True,
        output_version="v1",
        max_completion_tokens=10,
    )
    message = HumanMessage(
        content_blocks=[
            {
                "type": "text",
                "text": "Stable text prefix",
                "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
            },
            {
                "type": "image",
                "url": "https://example.com/image.png",
                "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
            },
            {
                "type": "file",
                "file_id": "file_123",
                "extras": {"prompt_cache_breakpoint": {"mode": "explicit"}},
            },
        ]
    )

    payload = chat._get_request_payload([message])

    assert payload["input"][0]["content"] == [
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


def test_prompt_cache_breakpoint_top_level_on_data_blocks() -> None:
    """A top-level `prompt_cache_breakpoint` on a data block is preserved."""
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "url": "https://example.com/image.png",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {
                "type": "file",
                "file_id": "file_123",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
        ]
    )

    payload = chat._get_request_payload([message])

    assert payload["messages"][0]["content"] == [
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


def test_prompt_cache_breakpoint_preserves_falsy_value() -> None:
    """A present-but-falsy breakpoint value is copied (membership, not truthiness)."""
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Stable prefix",
                "extras": {"prompt_cache_breakpoint": None},
            }
        ]
    )

    payload = chat._get_request_payload([message])

    block = payload["messages"][0]["content"][0]
    assert "prompt_cache_breakpoint" in block
    assert block["prompt_cache_breakpoint"] is None


def test_prompt_cache_breakpoint_text_block_drops_other_extras() -> None:
    """Promoting a breakpoint from `extras` drops the block's other `extras`."""
    chat = ChatOpenAI(model=MODEL_NAME, max_completion_tokens=10)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Stable prefix",
                "extras": {
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                    "unsupported": "value",
                },
            }
        ]
    )

    payload = chat._get_request_payload([message])

    assert payload["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Stable prefix",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
