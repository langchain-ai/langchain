"""Test MistralAI Chat API wrapper."""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)
from langchain_core.messages import content as types
from pydantic import SecretStr

from langchain_mistralai._compat import _convert_to_v1_from_mistral
from langchain_mistralai.chat_models import (  # type: ignore[import]
    ChatMistralAI,
    _convert_chunk_to_message_chunk,
    _convert_message_to_mistral_chat_message,
    _convert_mistral_chat_message_to_message,
    _convert_tool_call_id_to_mistral_compatible,
    _is_valid_mistral_tool_call_id,
)

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistralai_model_param() -> None:
    llm = ChatMistralAI(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


def test_mistralai_initialization() -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatMistralAI(model="test", mistral_api_key="test"),  # type: ignore[call-arg, call-arg]
        ChatMistralAI(model="test", api_key="test"),  # type: ignore[call-arg, arg-type]
    ]:
        assert cast("SecretStr", model.mistral_api_key).get_secret_value() == "test"


@pytest.mark.parametrize(
    ("model", "expected_url"),
    [
        (ChatMistralAI(model="test"), "https://api.mistral.ai/v1"),  # type: ignore[call-arg, arg-type]
        (ChatMistralAI(model="test", endpoint="baz"), "baz"),  # type: ignore[call-arg, arg-type]
    ],
)
def test_mistralai_initialization_baseurl(
    model: ChatMistralAI, expected_url: str
) -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized providing endpoint, but also
    # with default

    assert model.endpoint == expected_url


@pytest.mark.parametrize(
    "env_var_name",
    [
        ("MISTRAL_BASE_URL"),
    ],
)
def test_mistralai_initialization_baseurl_env(env_var_name: str) -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized using env variable
    import os

    os.environ[env_var_name] = "boo"
    model = ChatMistralAI(model="test")  # type: ignore[call-arg]
    assert model.endpoint == "boo"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="Hello"),
            {"role": "system", "content": "Hello"},
        ),
        (
            HumanMessage(content="Hello"),
            {"role": "user", "content": "Hello"},
        ),
        (
            AIMessage(content="Hello"),
            {"role": "assistant", "content": "Hello"},
        ),
        (
            AIMessage(content="{", additional_kwargs={"prefix": True}),
            {"role": "assistant", "content": "{", "prefix": True},
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            {"role": "assistant", "content": "Hello"},
        ),
    ],
)
def test_convert_message_to_mistral_chat_message(
    message: BaseMessage, expected: dict
) -> None:
    result = _convert_message_to_mistral_chat_message(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> dict:
    return {
        "id": "abc123",
        "model": "fake_model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None,
            }
        ],
    }


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    def it() -> Generator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    async def it() -> AsyncGenerator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


@patch(
    "langchain_mistralai.chat_models.ChatMistralAI.completion_with_retry",
    new=mock_chat_stream,
)
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch("langchain_mistralai.chat_models.acompletion_with_retry", new=mock_chat_astream)
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "ssAbar4Dr",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
    }
    message = {"role": "assistant", "content": "", "tool_calls": [raw_tool_call]}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="ssAbar4Dr",
                type="tool_call",
            )
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output
    assert _convert_message_to_mistral_chat_message(expected_output) == message

    # Test malformed tool call
    raw_tool_calls = [
        {
            "id": "pL5rEGzxe",
            "function": {
                "arguments": '{"name": "Sally", "hair_color": "green"}',
                "name": "GenerateUsername",
            },
        },
        {
            "id": "ssAbar4Dr",
            "function": {
                "arguments": "oops",
                "name": "GenerateUsername",
            },
        },
    ]
    message = {"role": "assistant", "content": "", "tool_calls": raw_tool_calls}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
                id="ssAbar4Dr",
                type="invalid_tool_call",
            ),
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="pL5rEGzxe",
                type="tool_call",
            ),
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output
    assert _convert_message_to_mistral_chat_message(expected_output) == message


def test__convert_dict_to_message_tool_call_with_null_content() -> None:
    raw_tool_call = {
        "id": "ssAbar4Dr",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="ssAbar4Dr",
                type="tool_call",
            )
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output


def test__convert_dict_to_message_with_missing_content() -> None:
    raw_tool_call = {
        "id": "ssAbar4Dr",
        "function": {
            "arguments": '{"query": "test search"}',
            "name": "search",
        },
    }
    message = {"role": "assistant", "tool_calls": [raw_tool_call]}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="search",
                args={"query": "test search"},
                id="ssAbar4Dr",
                type="tool_call",
            )
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output


def test__convert_dict_to_message_with_citations() -> None:
    """Reference blocks normalized to text blocks with reference metadata."""
    cited_text = "the temperature is 20 degrees C"
    raw_content: list[str | dict] = [
        {"type": "text", "text": "According to the document, "},
        {"type": "reference", "reference_ids": [0], "text": cited_text},
        {"type": "text", "text": " on average."},
    ]
    message = {"role": "assistant", "content": raw_content}
    result = _convert_mistral_chat_message_to_message(message)

    assert isinstance(result.content, list)
    content = result.content
    # The reference block is normalized to type="text" so .text includes it
    assert content[0] == {"type": "text", "text": "According to the document, "}
    assert isinstance(content[1], dict)
    block_1 = content[1]
    assert block_1["type"] == "text"
    assert block_1["text"] == cited_text
    assert block_1["reference"] == {"reference_ids": [0]}
    assert content[2] == {"type": "text", "text": " on average."}
    assert result.response_metadata["model_provider"] == "mistralai"
    assert "citations" not in result.response_metadata


def test__convert_dict_to_message_citations_text_accessor() -> None:
    """message.text includes cited spans from normalized reference blocks."""
    cited_text = "the temperature is 20 degrees C"
    raw_content: list[str | dict] = [
        {"type": "text", "text": "According to the document, "},
        {"type": "reference", "reference_ids": [0], "text": cited_text},
        {"type": "text", "text": " on average."},
    ]
    message = {"role": "assistant", "content": raw_content}
    result = _convert_mistral_chat_message_to_message(message)

    # .text should include all visible text, including the cited span
    assert str(result.text) == (
        "According to the document, the temperature is 20 degrees C on average."
    )


def test__convert_dict_to_message_citations_to_content_blocks() -> None:
    """content_blocks translates reference metadata to TextContentBlock."""
    cited_text = "the temperature is 20 degrees C"
    raw_content: list[str | dict] = [
        {"type": "text", "text": "According to the document, "},
        {"type": "reference", "reference_ids": [0], "text": cited_text},
        {"type": "text", "text": " on average."},
    ]
    message = {"role": "assistant", "content": raw_content}
    result = _convert_mistral_chat_message_to_message(message)

    assert isinstance(result, AIMessage)
    blocks = _convert_to_v1_from_mistral(result)
    assert len(blocks) == 3

    # First block: plain text
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "According to the document, "

    # Second block: text with citation annotation
    block_1 = cast("types.TextContentBlock", blocks[1])
    assert block_1["type"] == "text"
    assert block_1["text"] == cited_text
    annotations = block_1["annotations"]
    assert len(annotations) == 1
    assert annotations[0]["type"] == "citation"
    assert annotations[0]["cited_text"] == cited_text
    assert annotations[0]["extras"]["reference_ids"] == [0]

    # Third block: plain text
    assert blocks[2]["type"] == "text"
    assert blocks[2]["text"] == " on average."


def test_create_chat_result_with_citations() -> None:
    """Citations are normalized to text blocks with reference metadata in .content."""
    chat = ChatMistralAI()
    raw_citation = {"type": "reference", "reference_ids": [0], "text": "42"}
    raw_content: list[str | dict] = [
        {"type": "text", "text": "The answer is "},
        raw_citation,
        {"type": "text", "text": "."},
    ]
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": raw_content,
                },
                "finish_reason": "stop",
            }
        ]
    }

    result = chat._create_chat_result(response)
    message = result.generations[0].message

    assert isinstance(message.content, list)
    content = message.content
    # The reference block is normalized; .text includes the cited span
    assert isinstance(content[1], dict)
    block_1 = content[1]
    assert block_1["type"] == "text"
    assert block_1["text"] == "42"
    assert block_1["reference"] == {"reference_ids": [0]}
    assert str(message.text) == "The answer is 42."
    assert "citations" not in message.response_metadata


def test__convert_chunk_to_message_chunk_with_citations() -> None:
    """Streaming reference blocks are normalized to text blocks in chunk .content."""
    raw_citation = {"type": "reference", "reference_ids": [0], "text": "42"}
    text_chunk = {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "The answer is "},
                "finish_reason": None,
            }
        ],
    }
    reference_chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": [
                        dict(raw_citation),
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "model": "mistral-small-latest",
    }

    result_1, index, index_type = _convert_chunk_to_message_chunk(
        text_chunk, AIMessageChunk, -1, "", None
    )
    result_2, _, _ = _convert_chunk_to_message_chunk(
        reference_chunk, AIMessageChunk, index, index_type, None
    )

    assert isinstance(result_2, AIMessageChunk)
    # Reference block is normalized to type="text" with reference metadata
    assert result_2.content == [
        {"type": "text", "text": "42", "reference": {"reference_ids": [0]}, "index": 0},
    ]
    assert "citations" not in result_2.response_metadata

    full = result_1 + result_2
    assert isinstance(full, AIMessageChunk)
    assert "citations" not in full.response_metadata
    assert full.response_metadata["finish_reason"] == "stop"
    # .text includes the cited span
    assert str(full.text) == "The answer is 42"


def test_citation_round_trip() -> None:
    """Round-trip through v1 preserves text and reference metadata."""
    from langchain_mistralai._compat import (
        _convert_from_v1_to_mistral,
        _convert_to_v1_from_mistral,
    )

    # Start with normalized content (as produced by _convert_mistral_chat_message)
    original_content: list[str | dict] = [
        {"type": "text", "text": "The answer is "},
        {"type": "text", "text": "42", "reference": {"reference_ids": [0]}},
        {"type": "text", "text": "."},
    ]
    message = AIMessage(content=original_content)
    v1_blocks = _convert_to_v1_from_mistral(message)
    round_tripped = _convert_from_v1_to_mistral(v1_blocks, "mistralai")

    # Should have exactly 3 blocks, no duplication of cited text
    assert len(round_tripped) == 3
    assert round_tripped[0] == {"type": "text", "text": "The answer is "}
    assert isinstance(round_tripped[1], dict)
    block_1 = round_tripped[1]
    assert block_1["type"] == "text"
    assert block_1["text"] == "42"
    assert block_1["reference"] == {"reference_ids": [0]}
    assert round_tripped[2] == {"type": "text", "text": "."}


def test_citation_round_trip_preserves_extra_fields() -> None:
    """Extra provider fields on reference metadata survive the round-trip."""
    from langchain_mistralai._compat import (
        _convert_from_v1_to_mistral,
        _convert_to_v1_from_mistral,
    )

    original_content: list[str | dict] = [
        {"type": "text", "text": "cited span", "reference": {"reference_ids": [1, 2]}},
    ]
    message = AIMessage(content=original_content)
    v1_blocks = _convert_to_v1_from_mistral(message)
    round_tripped = _convert_from_v1_to_mistral(v1_blocks, "mistralai")

    assert len(round_tripped) == 1
    assert isinstance(round_tripped[0], dict)
    block_0 = round_tripped[0]
    assert block_0["type"] == "text"
    assert block_0["text"] == "cited span"
    assert block_0["reference"] == {"reference_ids": [1, 2]}


def test_citation_streaming_accumulated_content() -> None:
    """Streaming chunks accumulate normalized text blocks in full.content."""
    raw_citation = {"type": "reference", "reference_ids": [0], "text": "42"}
    text_chunk = {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "The answer is "},
                "finish_reason": None,
            }
        ],
    }
    reference_chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": [dict(raw_citation)],
                },
                "finish_reason": "stop",
            }
        ],
        "model": "mistral-small-latest",
    }

    result_1, index, index_type = _convert_chunk_to_message_chunk(
        text_chunk, AIMessageChunk, -1, "", None
    )
    result_2, _, _ = _convert_chunk_to_message_chunk(
        reference_chunk, AIMessageChunk, index, index_type, None
    )

    full = result_1 + result_2
    # full.content should contain both the text and the normalized reference block
    assert isinstance(full.content, list)
    assert any(
        isinstance(b, dict)
        and b.get("type") == "text"
        and b.get("text") == "42"
        and isinstance(ref := b.get("reference"), dict)
        and ref.get("reference_ids") == [0]
        for b in full.content
    )


def test_citation_index_not_in_extras() -> None:
    """Streaming index should not leak into citation extras."""
    from langchain_mistralai._compat import _convert_to_v1_from_mistral

    content: list[str | dict] = [
        {"type": "text", "text": "42", "reference": {"reference_ids": [0]}, "index": 0},
    ]
    message = AIMessageChunk(content=content)
    blocks = _convert_to_v1_from_mistral(message)
    assert len(blocks) == 1
    block_0 = cast("types.TextContentBlock", blocks[0])
    annotation = block_0["annotations"][0]
    extras = annotation.get("extras", {})
    assert isinstance(extras, dict)
    assert "index" not in extras


def test_citation_no_text_in_reference() -> None:
    """A reference block with no text still converts without error."""
    from langchain_mistralai._compat import _convert_to_v1_from_mistral

    content: list[str | dict] = [
        {"type": "text", "text": "", "reference": {"reference_ids": [0]}},
    ]
    message = AIMessage(content=content)
    blocks = _convert_to_v1_from_mistral(message)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == ""
    block_0 = cast("types.TextContentBlock", blocks[0])
    assert "cited_text" not in block_0["annotations"][0]


def test_malformed_annotation_does_not_crash() -> None:
    """Malformed annotations are skipped, not raised."""
    from langchain_mistralai._compat import _convert_from_v1_to_mistral

    content: list = [
        {
            "type": "text",
            "text": "hello",
            "annotations": [
                None,  # not a dict
                {"type": "unknown"},  # unrecognized type
                {"type": "citation", "cited_text": "cited"},  # valid
            ],
        }
    ]
    result = _convert_from_v1_to_mistral(content, "mistralai")
    # The valid citation produces a text block with reference metadata;
    # the text block is not appended because a reference was emitted.
    assert len(result) == 1
    assert isinstance(result[0], dict)
    block_0 = result[0]
    assert block_0["type"] == "text"
    assert block_0["text"] == "cited"
    assert "reference" in block_0


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> list[int]:
        return [1, 2, 3]

    llm = ChatMistralAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]


def test_tool_id_conversion() -> None:
    assert _is_valid_mistral_tool_call_id("ssAbar4Dr")
    assert not _is_valid_mistral_tool_call_id("abc123")
    assert not _is_valid_mistral_tool_call_id("call_JIIjI55tTipFFzpcP8re3BpM")

    result_map = {
        "ssAbar4Dr": "ssAbar4Dr",
        "abc123": "pL5rEGzxe",
        "call_JIIjI55tTipFFzpcP8re3BpM": "8kxAQvoED",
    }
    for input_id, expected_output in result_map.items():
        assert _convert_tool_call_id_to_mistral_compatible(input_id) == expected_output
        assert _is_valid_mistral_tool_call_id(expected_output)


def test_extra_kwargs() -> None:
    # Check that foo is saved in extra_kwargs.
    llm = ChatMistralAI(model="my-model", foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatMistralAI(model="my-model", foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatMistralAI(model="my-model", foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]


def test_retry_with_failure_then_success() -> None:
    """Test retry mechanism works correctly when fiest request fails, second succeed."""
    # Create a real ChatMistralAI instance
    chat = ChatMistralAI(max_retries=3)

    # Set up the actual retry mechanism (not just mocking it)
    # We'll track how many times the function is called
    call_count = 0

    def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            msg = "Connection error"
            raise httpx.RequestError(msg, request=MagicMock())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return mock_response

    with patch.object(chat.client, "post", side_effect=mock_post):
        result = chat.invoke("Hello")
        assert result.content == "Hello!"
        assert call_count == 2, f"Expected 2 calls, but got {call_count}"


def test_no_duplicate_tool_calls_when_multiple_tools() -> None:
    """
    Tests whether the conversion of an AIMessage with more than one tool call
    to a Mistral assistant message correctly returns each tool call exactly
    once in the final payload.

    The current implementation uses a faulty for loop which produces N*N entries in the
    final tool_calls array of the payload (and thus duplicates tool call ids).
    """
    msg = AIMessage(
        content="",  # content should be blank when tool_calls are present
        tool_calls=[
            ToolCall(name="tool_a", args={"x": 1}, id="id_a", type="tool_call"),
            ToolCall(name="tool_b", args={"y": 2}, id="id_b", type="tool_call"),
        ],
        response_metadata={"model_provider": "mistralai"},
    )

    mistral_msg = _convert_message_to_mistral_chat_message(msg)

    assert mistral_msg["role"] == "assistant"
    assert "tool_calls" in mistral_msg, "Expected tool_calls to be present."

    tool_calls = mistral_msg["tool_calls"]
    # With the bug, this would be 4 (2x2); we expect exactly 2 entries.
    assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

    # Ensure there are no duplicate ids
    ids = [tc.get("id") for tc in tool_calls if isinstance(tc, dict)]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"Duplicate tool call IDs found: {ids}"


def test_profile() -> None:
    model = ChatMistralAI(model="mistral-large-latest")  # type: ignore[call-arg]
    assert model.profile
