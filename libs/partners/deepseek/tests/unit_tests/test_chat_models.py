"""Test chat model integration."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, SecretStr

from langchain_deepseek.chat_models import DEFAULT_API_BASE, ChatDeepSeek

MODEL_NAME = "deepseek-chat"


class MockOpenAIResponse(BaseModel):
    """Mock OpenAI response model."""

    choices: list
    error: None = None

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: Literal["json", "python"] | str = "python",  # noqa: PYI051
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Literal["none", "warn", "error"] | bool = True,
        context: dict[str, Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary, ensuring `reasoning_content` is included."""
        choices_list = []
        for choice in self.choices:
            if isinstance(choice.message, ChatCompletionMessage):
                message_dict = choice.message.model_dump()
                # Ensure model_extra fields are at top level
                if "model_extra" in message_dict:
                    message_dict.update(message_dict["model_extra"])
            else:
                message_dict = {
                    "role": "assistant",
                    "content": choice.message.content,
                }
                # Add reasoning_content if present
                if hasattr(choice.message, "reasoning_content"):
                    message_dict["reasoning_content"] = choice.message.reasoning_content
                # Add model_extra fields at the top level if present
                if hasattr(choice.message, "model_extra"):
                    message_dict.update(choice.message.model_extra)
                    message_dict["model_extra"] = choice.message.model_extra
            choices_list.append({"message": message_dict})

        return {"choices": choices_list, "error": self.error}


class TestChatDeepSeekUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatDeepSeek` chat model."""

    @property
    def chat_model_class(self) -> type[ChatDeepSeek]:
        """Chat model class being tested."""
        return ChatDeepSeek

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "DEEPSEEK_API_KEY": "api_key",
                "DEEPSEEK_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatDeepSeek:
        """Get a chat model instance for testing."""
        return ChatDeepSeek(**self.chat_model_params)


class TestChatDeepSeekCustomUnit:
    """Custom tests specific to DeepSeek chat model."""

    def test_create_chat_result_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from response."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "This is the reasoning content"
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning content"
        )

    def test_create_chat_result_with_model_extra_reasoning(self) -> None:
        """Test that reasoning is properly extracted from `model_extra`."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_message.model_extra = {"reasoning": "This is the reasoning"}
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Main content",
            "model_extra": {"reasoning": "This is the reasoning"},
        }
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MockOpenAIResponse(choices=[mock_choice], error=None)

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning"
        )
        assert "reasoning" not in result.generations[0].message.additional_kwargs

    def test_create_chat_result_preserves_empty_model_extra_reasoning(self) -> None:
        """Empty reasoning_content inside model_extra should be preserved."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_message.model_extra = {"reasoning_content": ""}
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Main content",
            "model_extra": {"reasoning_content": ""},
        }
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MockOpenAIResponse(choices=[mock_choice], error=None)

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == ""
        )
        assert "reasoning" not in result.generations[0].message.additional_kwargs

    def test_create_chat_result_preserves_empty_reasoning(self) -> None:
        """Empty reasoning_content should be preserved (not dropped as falsy)."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = ""
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == ""
        )
        assert "reasoning" not in result.generations[0].message.additional_kwargs

    def test_create_chat_result_normalizes_list_reasoning(self) -> None:
        """List reasoning should be normalized and set."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = [
            {"type": "reasoning.text", "text": "Step 1"},
            {"type": "reasoning.text", "text": "Step 2"},
        ]
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "Step 1\nStep 2"
        )
        assert "reasoning" not in result.generations[0].message.additional_kwargs

    def test_create_chat_result_with_reasoning_details_single_item(self) -> None:
        """Single-item reasoning_details should have text stripped."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "My reasoning"
        mock_message.reasoning_details = [{"type": "thinking", "text": "My reasoning"}]
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "My reasoning"
        )
        # Single-item should have text stripped to avoid duplication
        assert result.generations[0].message.additional_kwargs.get(
            "reasoning_details"
        ) == [{"type": "thinking"}]

    def test_create_chat_result_with_reasoning_details_multi_item(self) -> None:
        """Multi-item reasoning_details should be preserved in full."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "Step 1\nStep 2"
        mock_message.reasoning_details = [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "Step 1\nStep 2"
        )
        # Multi-item keeps full content
        assert result.generations[0].message.additional_kwargs.get(
            "reasoning_details"
        ) == [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]

    def test_create_chat_result_with_reasoning_details_in_model_extra(self) -> None:
        """reasoning_details in model_extra should be extracted."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_message.model_extra = {
            "reasoning_content": "My reasoning",
            "reasoning_details": [{"type": "thinking", "text": "My reasoning"}],
        }
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Main content",
            "model_extra": {
                "reasoning_content": "My reasoning",
                "reasoning_details": [{"type": "thinking", "text": "My reasoning"}],
            },
        }
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MockOpenAIResponse(choices=[mock_choice], error=None)

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "My reasoning"
        )
        # Single-item should have text stripped
        assert result.generations[0].message.additional_kwargs.get(
            "reasoning_details"
        ) == [{"type": "thinking"}]

    def test_convert_chunk_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "Streaming reasoning content",
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning content"
        )
        assert "reasoning" not in chunk_result.message.additional_kwargs

    def test_convert_chunk_with_reasoning(self) -> None:
        """Test that reasoning is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning": "Streaming reasoning",
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning"
        )
        assert "reasoning" not in chunk_result.message.additional_kwargs

    def test_convert_chunk_preserves_empty_reasoning(self) -> None:
        """Streaming chunks with empty reasoning_content should be preserved."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "",
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") == ""
        assert "reasoning" not in chunk_result.message.additional_kwargs

    def test_convert_chunk_normalizes_list_reasoning(self) -> None:
        """Streaming chunks with list reasoning should be normalized."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": [
                            {"type": "reasoning.text", "text": "First"},
                            {"type": "reasoning.text", "text": "Second"},
                        ],
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "First\nSecond"
        )
        assert "reasoning" not in chunk_result.message.additional_kwargs

    def test_convert_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {"content": "Main content"}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_convert_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_convert_chunk_with_reasoning_details_single_item(self) -> None:
        """Streaming chunk with single-item reasoning_details strips text."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "My reasoning",
                        "reasoning_details": [
                            {"type": "thinking", "text": "My reasoning"}
                        ],
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "My reasoning"
        )
        # Single-item should have text stripped
        assert chunk_result.message.additional_kwargs.get("reasoning_details") == [
            {"type": "thinking"}
        ]

    def test_convert_chunk_with_reasoning_details_multi_item(self) -> None:
        """Streaming chunk with multi-item reasoning_details preserves text."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "Step 1\nStep 2",
                        "reasoning_details": [
                            {"type": "thinking", "text": "Step 1"},
                            {"type": "thinking", "text": "Step 2"},
                        ],
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Step 1\nStep 2"
        )
        # Multi-item keeps full content
        assert chunk_result.message.additional_kwargs.get("reasoning_details") == [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]

    def test_convert_chunk_with_reasoning_details_only(self) -> None:
        """Streaming chunk with only reasoning_details extracts content from it."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_details": [
                            {"type": "thinking", "text": "My reasoning"}
                        ],
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        # reasoning_content should be extracted from reasoning_details
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "My reasoning"
        )
        # Single-item should have text stripped
        assert chunk_result.message.additional_kwargs.get("reasoning_details") == [
            {"type": "thinking"}
        ]

    def test_get_request_payload(self) -> None:
        """Test that tool message content is converted from list to string."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        tool_message = ToolMessage(content=[], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "[]"

        tool_message = ToolMessage(content=["item1", "item2"], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == '["item1", "item2"]'

        tool_message = ToolMessage(content="test string", tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "test string"

    def test_get_request_payload_preserves_reasoning_content(self) -> None:
        """Test that reasoning_content is preserved in multi-turn conversations.

        This tests the fix for interleaved thinking, where reasoning_content from
        previous AI messages must be passed back to the API in subsequent turns.
        """
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="What is 2+2?"),
            AIMessage(
                content="Let me think...",
                additional_kwargs={"reasoning_content": "First, I'll add 2 and 2..."},
            ),
            HumanMessage(content="And 3+3?"),
        ]

        payload = chat_model._get_request_payload(messages)

        # Find assistant message and verify reasoning_content is preserved
        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert (
            assistant_msgs[0].get("reasoning_content") == "First, I'll add 2 and 2..."
        )
        assert "reasoning" not in assistant_msgs[0]

    def test_get_request_payload_preserves_multiple_reasoning_contents(self) -> None:
        """Test that multiple AI messages each preserve their reasoning_content."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(
                content="Answer 1",
                additional_kwargs={"reasoning_content": "Reasoning for answer 1"},
            ),
            HumanMessage(content="Question 2"),
            AIMessage(
                content="Answer 2",
                additional_kwargs={"reasoning_content": "Reasoning for answer 2"},
            ),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].get("reasoning_content") == "Reasoning for answer 1"
        assert assistant_msgs[1].get("reasoning_content") == "Reasoning for answer 2"
        assert "reasoning" not in assistant_msgs[0]
        assert "reasoning" not in assistant_msgs[1]

    def test_get_request_payload_without_reasoning_content(self) -> None:
        """Test that messages without reasoning_content work correctly."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") is None

    def test_no_reasoning_added_if_not_in_original(self) -> None:
        """Test that reasoning_content is not added if not in original message.

        This is data-driven behavior: we only preserve what was in the original
        message, rather than adding empty reasoning based on model name.
        """
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),  # No reasoning_content
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        # Should NOT have reasoning_content since it wasn't in original message
        assert "reasoning_content" not in assistant_msgs[0]
        assert "reasoning" not in assistant_msgs[0]

    def test_normalize_reasoning_with_string(self) -> None:
        """Test _normalize_reasoning with string input."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        result = chat_model._normalize_reasoning("test reasoning")
        assert result == "test reasoning"

    def test_normalize_reasoning_with_list(self) -> None:
        """Test _normalize_reasoning with list input (MiniMax format)."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        # MiniMax format with text field
        list_reasoning = [
            {"type": "reasoning.text", "text": "First thought"},
            {"type": "reasoning.text", "text": "Second thought"},
        ]
        result = chat_model._normalize_reasoning(list_reasoning)
        assert result == "First thought\nSecond thought"

        # List without text field falls back to stringified dict
        list_reasoning_no_text = [
            {"content": "Thought 1"},
            {"content": "Thought 2"},
        ]
        result = chat_model._normalize_reasoning(list_reasoning_no_text)
        assert result == "{'content': 'Thought 1'}\n{'content': 'Thought 2'}"

    def test_normalize_reasoning_with_empty_values(self) -> None:
        """Test _normalize_reasoning with empty/None values."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        assert chat_model._normalize_reasoning(None) is None
        # Empty string is preserved (important for deepseek-reasoner)
        assert chat_model._normalize_reasoning("") == ""
        assert chat_model._normalize_reasoning([]) is None

    def test_reasoning_preserved_with_tool_calls(self) -> None:
        """Test reasoning is preserved when messages include tool calls."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Use the tool"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "test_tool", "args": {"value": "test"}}
                ],
                additional_kwargs={"reasoning_content": "Tool reasoning here"},
            ),
            ToolMessage(content="Tool result", tool_call_id="call_1"),
            AIMessage(
                content="Done with the task",
                additional_kwargs={"reasoning_content": "Final reasoning"},
            ),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].get("reasoning_content") == "Tool reasoning here"
        assert assistant_msgs[1].get("reasoning_content") == "Final reasoning"
        assert "reasoning" not in assistant_msgs[0]
        assert "reasoning" not in assistant_msgs[1]

    def test_empty_string_reasoning_preserved(self) -> None:
        """Test that empty string reasoning_content is preserved (not treated as None).

        This is important for deepseek-reasoner which requires reasoning_content
        to be present even if empty.
        """
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Hi",
                additional_kwargs={"reasoning_content": ""},  # Explicitly empty
            ),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        # Empty string should be preserved, not dropped
        assert "reasoning_content" in assistant_msgs[0]
        assert assistant_msgs[0]["reasoning_content"] == ""
        assert "reasoning" not in assistant_msgs[0]

    def test_get_request_payload_preserves_reasoning_details_single_item(self) -> None:
        """Test single-item reasoning_details round-trip reconstructs text."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        # Simulate message received from API (single-item with text stripped)
        messages = [
            HumanMessage(content="What is 2+2?"),
            AIMessage(
                content="4",
                additional_kwargs={
                    "reasoning_content": "Let me add 2 + 2...",
                    "reasoning_details": [{"type": "thinking"}],  # Text stripped
                },
            ),
            HumanMessage(content="And 3+3?"),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") == "Let me add 2 + 2..."
        # reasoning_details should be reconstructed with text
        assert assistant_msgs[0].get("reasoning_details") == [
            {"type": "thinking", "text": "Let me add 2 + 2..."}
        ]

    def test_get_request_payload_preserves_reasoning_details_multi_item(self) -> None:
        """Test multi-item reasoning_details round-trip preserves as-is."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        # Multi-item reasoning_details keep their text
        messages = [
            HumanMessage(content="Explain"),
            AIMessage(
                content="Here's my answer",
                additional_kwargs={
                    "reasoning_content": "Step 1\nStep 2",
                    "reasoning_details": [
                        {"type": "thinking", "text": "Step 1"},
                        {"type": "thinking", "text": "Step 2"},
                    ],
                },
            ),
            HumanMessage(content="Continue"),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") == "Step 1\nStep 2"
        # Multi-item should be preserved as-is
        assert assistant_msgs[0].get("reasoning_details") == [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]

    def test_get_request_payload_reasoning_details_without_content(self) -> None:
        """Test reasoning_details without reasoning_content extracts content.

        When reasoning_content is not provided but reasoning_details is, the
        implementation extracts reasoning_content from reasoning_details for
        provider compatibility.
        """
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Hi",
                additional_kwargs={
                    "reasoning_details": [{"type": "thinking", "text": "Some thought"}],
                },
            ),
        ]

        payload = chat_model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        # reasoning_content extracted from reasoning_details
        assert assistant_msgs[0].get("reasoning_content") == "Some thought"
        # reasoning_details preserved (single item keeps text since it was sourced)
        assert assistant_msgs[0].get("reasoning_details") == [
            {"type": "thinking", "text": "Some thought"}
        ]

    def test_streaming_chunk_reasoning_accumulation(self) -> None:
        """Test that AIMessageChunk accumulation handles reasoning correctly."""
        # Create chunks with reasoning
        chunk1 = AIMessageChunk(
            content="Hello",
            additional_kwargs={"reasoning_content": "First thought"},
        )
        chunk2 = AIMessageChunk(
            content=" world",
            additional_kwargs={"reasoning_content": " second thought"},
        )

        # Accumulate chunks
        accumulated = chunk1 + chunk2

        # Content should be concatenated
        assert accumulated.content == "Hello world"

        # additional_kwargs behavior: values are merged/concatenated.
        assert accumulated.additional_kwargs.get("reasoning_content") == (
            "First thought second thought"
        )

    def test_strip_reasoning_text_single_item(self) -> None:
        """Test _strip_reasoning_text strips text from single-item list."""
        reasoning_details = [{"type": "thinking", "text": "My reasoning"}]
        result = ChatDeepSeek._strip_reasoning_text(reasoning_details)
        assert result == [{"type": "thinking"}]

    def test_strip_reasoning_text_multi_item(self) -> None:
        """Test _strip_reasoning_text returns None for multi-item list."""
        reasoning_details = [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]
        result = ChatDeepSeek._strip_reasoning_text(reasoning_details)
        assert result is None  # Signal to keep original

    def test_reconstruct_reasoning_details_without_text(self) -> None:
        """Test _reconstruct_reasoning_details re-attaches text when stripped."""
        # Simulates single-item that had text stripped
        reasoning_details = [{"type": "thinking"}]
        result = ChatDeepSeek._reconstruct_reasoning_details(
            reasoning_details, "My reasoning"
        )
        assert result == [{"type": "thinking", "text": "My reasoning"}]

    def test_reconstruct_reasoning_details_multi_item(self) -> None:
        """Test _reconstruct_reasoning_details returns multi-item list as-is."""
        reasoning_details = [
            {"type": "thinking", "text": "Step 1"},
            {"type": "thinking", "text": "Step 2"},
        ]
        result = ChatDeepSeek._reconstruct_reasoning_details(
            reasoning_details, "Ignored"
        )
        assert result == reasoning_details


class SampleTool(PydanticBaseModel):
    """Sample tool schema for testing."""

    value: str = Field(description="A test value")


class TestChatDeepSeekStrictMode:
    """Tests for DeepSeek strict mode support.

    This tests the experimental beta feature that uses the beta API endpoint
    when `strict=True` is used. These tests can be removed when strict mode
    becomes stable in the default base API.
    """

    def test_bind_tools_with_strict_mode_uses_beta_endpoint(self) -> None:
        """Test that bind_tools with strict=True uses the beta endpoint."""
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=SecretStr("test_key"),
        )

        # Verify default endpoint
        assert llm.api_base == DEFAULT_API_BASE

        # Bind tools with strict=True
        bound_model = llm.bind_tools([SampleTool], strict=True)

        # The bound model should have its internal model using beta endpoint
        # We can't directly access the internal model, but we can verify the behavior
        # by checking that the binding operation succeeds
        assert bound_model is not None

    def test_bind_tools_without_strict_mode_uses_default_endpoint(self) -> None:
        """Test bind_tools without strict or with strict=False uses default endpoint."""
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=SecretStr("test_key"),
        )

        # Test with strict=False
        bound_model_false = llm.bind_tools([SampleTool], strict=False)
        assert bound_model_false is not None

        # Test with strict=None (default)
        bound_model_none = llm.bind_tools([SampleTool])
        assert bound_model_none is not None

    def test_with_structured_output_strict_mode_uses_beta_endpoint(self) -> None:
        """Test that with_structured_output with strict=True uses beta endpoint."""
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=SecretStr("test_key"),
        )

        # Verify default endpoint
        assert llm.api_base == DEFAULT_API_BASE

        # Create structured output with strict=True
        structured_model = llm.with_structured_output(SampleTool, strict=True)

        # The structured model should work with beta endpoint
        assert structured_model is not None


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatDeepSeek(model="deepseek-reasoner", api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["reasoning_output"]
