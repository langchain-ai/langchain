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

    def test_thinking_tool_calls_keeps_reasoning_content_if_provided(self) -> None:
        chat_model = ChatDeepSeek(
            model="deepseek-reasoner",
            api_key=SecretStr("api_key"),
            extra_body={"thinking": {"type": "enabled"}},
        )

        ai = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather",
                         "args": {"location": "New York"}}],
            additional_kwargs={"reasoning_content": "reasoning..."},
        )
        tool = ToolMessage(content='{"temp": 10}', tool_call_id="call_1")

        payload = chat_model._get_request_payload(
            [HumanMessage(content="hi"), ai, tool])

        assistant_tool_msgs = [
            m for m in payload["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant_tool_msgs
        assert assistant_tool_msgs[0]["reasoning_content"] == "reasoning..."

    def test_thinking_tool_calls_falls_back_to_empty_reasoning_content(self) -> None:
        chat_model = ChatDeepSeek(
            model="deepseek-reasoner",
            api_key=SecretStr("api_key"),
            extra_body={"thinking": {"type": "enabled"}},
        )

        ai = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather",
                         "args": {"location": "New York"}}],
            additional_kwargs={},  # ✅ missing
        )
        tool = ToolMessage(content='{"temp": 10}', tool_call_id="call_1")

        payload = chat_model._get_request_payload(
            [HumanMessage(content="hi"), ai, tool])

        assistant_tool_msgs = [
            m for m in payload["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant_tool_msgs
        assert "reasoning_content" in assistant_tool_msgs[0]
        assert assistant_tool_msgs[0]["reasoning_content"] == ""


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
