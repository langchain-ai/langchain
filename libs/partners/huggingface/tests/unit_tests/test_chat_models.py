from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatResult, Generation, LLMResult
from langchain_core.tools import BaseTool

from langchain_huggingface.chat_models import (  # type: ignore[import]
    ChatHuggingFace,
    _convert_dict_to_message,
)
from langchain_huggingface.chat_models.huggingface import (
    _parse_tool_calls_from_text,
)
from langchain_huggingface.llms import HuggingFaceEndpoint


@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.inference_server_url = "test endpoint url"
    llm.temperature = 0.7
    llm.max_new_tokens = 512
    llm.top_p = 0.9
    llm.seed = 42
    llm.streaming = True
    llm.repetition_penalty = 1.1
    llm.stop_sequences = ["</s>", "<|end|>"]
    llm.model_kwargs = {"do_sample": True, "top_k": 50}
    llm.server_kwargs = {"timeout": 120}
    llm.repo_id = "test/model"
    llm.model = "test/model"
    return llm


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)
def chat_hugging_face(mock_resolve_id: Any, mock_llm: Any) -> ChatHuggingFace:
    return ChatHuggingFace(llm=mock_llm, tokenizer=MagicMock())


def test_create_chat_result(chat_hugging_face: Any) -> None:
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "test message"},
                "finish_reason": "test finish reason",
            }
        ],
        "usage": {"tokens": 420},
    }

    result = chat_hugging_face._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"  # type: ignore[index]
    )
    assert result.llm_output["token_usage"]["tokens"] == 420  # type: ignore[index]
    assert result.llm_output["model_name"] == chat_hugging_face.model_id  # type: ignore[index]


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "At least one HumanMessage must be provided!"),
        (
            [HumanMessage(content="Hi"), AIMessage(content="Hello")],
            "Last message must be a HumanMessage or ToolMessage!",
        ),
    ],
)
def test_to_chat_prompt_errors(
    chat_hugging_face: Any, messages: list[BaseMessage], expected_error: str
) -> None:
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chat_prompt(messages)
    assert expected_error in str(e.value)


def test_to_chat_prompt_valid_messages(chat_hugging_face: Any) -> None:
    messages = [AIMessage(content="Hello"), HumanMessage(content="How are you?")]
    expected_prompt = "Generated chat prompt"

    chat_hugging_face.tokenizer.apply_chat_template.return_value = expected_prompt

    result = chat_hugging_face._to_chat_prompt(messages)

    assert result == expected_prompt
    chat_hugging_face.tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="You are a helpful assistant."),
            {"role": "system", "content": "You are a helpful assistant."},
        ),
        (
            AIMessage(content="How can I help you?"),
            {"role": "assistant", "content": "How can I help you?"},
        ),
        (
            HumanMessage(content="Hello"),
            {"role": "user", "content": "Hello"},
        ),
    ],
)
def test_to_chatml_format(
    chat_hugging_face: Any, message: BaseMessage, expected: dict[str, str]
) -> None:
    result = chat_hugging_face._to_chatml_format(message)
    assert result == expected


def test_to_chatml_format_with_invalid_type(chat_hugging_face: Any) -> None:
    message = "Invalid message type"
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chatml_format(message)
    assert "Unknown message type:" in str(e.value)


@pytest.mark.parametrize(
    ("msg_dict", "expected_type", "expected_content"),
    [
        (
            {"role": "system", "content": "You are helpful"},
            SystemMessage,
            "You are helpful",
        ),
        (
            {"role": "user", "content": "Hello there"},
            HumanMessage,
            "Hello there",
        ),
        (
            {"role": "assistant", "content": "How can I help?"},
            AIMessage,
            "How can I help?",
        ),
        (
            {"role": "function", "content": "result", "name": "get_time"},
            FunctionMessage,
            "result",
        ),
    ],
)
def test_convert_dict_to_message(
    msg_dict: dict[str, Any], expected_type: type, expected_content: str
) -> None:
    result = _convert_dict_to_message(msg_dict)
    assert isinstance(result, expected_type)
    assert result.content == expected_content


def tool_mock() -> dict:
    return {"function": {"name": "test_tool"}}


@pytest.mark.parametrize(
    "tools, tool_choice, expected_exception, expected_message",
    [
        ([tool_mock()], ["invalid type"], ValueError, "Unrecognized tool_choice type."),
        (
            [tool_mock(), tool_mock()],
            "test_tool",
            ValueError,
            "must provide exactly one tool.",
        ),
        (
            [tool_mock()],
            {"type": "function", "function": {"name": "other_tool"}},
            ValueError,
            "Tool choice {'type': 'function', 'function': {'name': 'other_tool'}} "
            "was specified, but the only provided tool was test_tool.",
        ),
    ],
)
def test_bind_tools_errors(
    chat_hugging_face: Any,
    tools: dict[str, str],
    tool_choice: Any,
    expected_exception: Any,
    expected_message: str,
) -> None:
    with patch(
        "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
        side_effect=lambda x: x,
    ):
        with pytest.raises(expected_exception) as excinfo:
            chat_hugging_face.bind_tools(tools, tool_choice=tool_choice)
        assert expected_message in str(excinfo.value)


def test_bind_tools(chat_hugging_face: Any) -> None:
    tools = [MagicMock(spec=BaseTool)]
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
            side_effect=lambda x: x,
        ),
        patch("langchain_core.runnables.base.Runnable.bind") as mock_super_bind,
    ):
        chat_hugging_face.bind_tools(tools, tool_choice="auto")
        mock_super_bind.assert_called_once()
        _, kwargs = mock_super_bind.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


def test_property_inheritance_integration(chat_hugging_face: Any) -> None:
    """Test that ChatHuggingFace inherits params from LLM object."""
    assert getattr(chat_hugging_face, "temperature", None) == 0.7
    assert getattr(chat_hugging_face, "max_tokens", None) == 512
    assert getattr(chat_hugging_face, "top_p", None) == 0.9
    assert getattr(chat_hugging_face, "streaming", None) is True


def test_default_params_includes_inherited_values(chat_hugging_face: Any) -> None:
    """Test that _default_params includes inherited max_tokens from max_new_tokens."""
    params = chat_hugging_face._default_params
    assert params["max_tokens"] == 512  # inherited from LLM's max_new_tokens
    assert params["temperature"] == 0.7  # inherited from LLM's temperature
    assert params["stream"] is True  # inherited from LLM's streaming


def test_create_message_dicts_includes_inherited_params(chat_hugging_face: Any) -> None:
    """Test that _create_message_dicts includes inherited parameters in API call."""
    messages = [HumanMessage(content="test message")]
    message_dicts, params = chat_hugging_face._create_message_dicts(messages, None)

    # Verify inherited parameters are included
    assert params["max_tokens"] == 512
    assert params["temperature"] == 0.7
    assert params["stream"] is True

    # Verify message conversion
    assert len(message_dicts) == 1
    assert message_dicts[0]["role"] == "user"
    assert message_dicts[0]["content"] == "test message"


def test_model_kwargs_inheritance(mock_llm: Any) -> None:
    """Test that model_kwargs are inherited when not explicitly set."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert chat.model_kwargs == {"do_sample": True, "top_k": 50}


def test_huggingface_endpoint_specific_inheritance(mock_llm: Any) -> None:
    """Test HuggingFaceEndpoint specific parameter inheritance."""
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
        ),
        patch(
            "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
            return_value=True,
        ),
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert (
            getattr(chat, "frequency_penalty", None) == 1.1
        )  # from repetition_penalty


def test_parameter_precedence_explicit_over_inherited(mock_llm: Any) -> None:
    """Test that explicitly set parameters take precedence over inherited ones."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Explicitly set max_tokens to override inheritance
        chat = ChatHuggingFace(llm=mock_llm, max_tokens=256, temperature=0.5)
        assert chat.max_tokens == 256  # explicit value, not inherited 512
        assert chat.temperature == 0.5  # explicit value, not inherited 0.7


def test_inheritance_with_no_llm_properties(mock_llm: Any) -> None:
    """Test inheritance when LLM doesn't have expected properties."""
    # Remove some properties from mock
    del mock_llm.temperature
    del mock_llm.top_p

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        # Should still inherit available properties
        assert chat.max_tokens == 512  # max_new_tokens still available
        # Missing properties should remain None/default
        assert getattr(chat, "temperature", None) is None
        assert getattr(chat, "top_p", None) is None


def test_inheritance_with_empty_llm() -> None:
    """Test that inheritance handles LLM with no relevant attributes gracefully."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Create a minimal mock LLM that passes validation but has no
        # inheritance attributes
        empty_llm = Mock(spec=HuggingFaceEndpoint)
        empty_llm.repo_id = "test/model"
        empty_llm.model = "test/model"
        # Mock doesn't have the inheritance attributes by default

        chat = ChatHuggingFace(llm=empty_llm)
        # Properties should remain at their default values when LLM has no
        # relevant attrs
        assert chat.max_tokens is None
        assert chat.temperature is None


def test_profile() -> None:
    empty_llm = Mock(spec=HuggingFaceEndpoint)
    empty_llm.repo_id = "test/model"
    empty_llm.model = "test/model"

    model = ChatHuggingFace(
        model_id="moonshotai/Kimi-K2-Instruct-0905",
        llm=empty_llm,
    )
    assert model.profile


def test_init_chat_model_huggingface() -> None:
    """Test that init_chat_model works with HuggingFace models.

    This test verifies that the fix for issue #28226 works correctly.
    The issue was that init_chat_model didn't properly handle HuggingFace
    model initialization, particularly the required 'task' parameter and
    parameter separation between HuggingFacePipeline and ChatHuggingFace.
    """
    from langchain.chat_models.base import init_chat_model

    # Test basic initialization with default task
    # Note: This test may skip in CI if model download fails, but it verifies
    # that the initialization code path works correctly
    try:
        llm = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            temperature=0,
            max_tokens=1024,
        )

        # Verify that ChatHuggingFace was created successfully
        assert llm is not None
        from langchain_huggingface import ChatHuggingFace

        assert isinstance(llm, ChatHuggingFace)

        # Verify that the llm attribute is set (this was the bug - it was missing)
        assert hasattr(llm, "llm")
        assert llm.llm is not None

        # Test with explicit task parameter
        llm2 = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            task="text-generation",
            temperature=0.5,
        )
        assert isinstance(llm2, ChatHuggingFace)
        assert llm2.llm is not None
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as e:
        # If model download fails in CI, skip the test rather than failing
        # The important part is that the code path doesn't raise ValidationError
        # about missing 'llm' field, which was the original bug
        pytest.skip(f"Skipping test due to model download/initialization error: {e}")


# ---------------------------------------------------------------------------
# Tool-calling support tests (pipeline path)
# ---------------------------------------------------------------------------


class TestToChatmlFormatToolMessages:
    """Test _to_chatml_format with tool-related message types."""

    def test_tool_message(self, chat_hugging_face: Any) -> None:
        msg = ToolMessage(content="42", tool_call_id="call_abc123")
        result = chat_hugging_face._to_chatml_format(msg)
        assert result == {
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_abc123",
        }

    def test_ai_message_with_tool_calls(self, chat_hugging_face: Any) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="get_weather", args={"city": "Paris"}, id="call_1"),
            ],
        )
        result = chat_hugging_face._to_chatml_format(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_ai_message_without_tool_calls(self, chat_hugging_face: Any) -> None:
        msg = AIMessage(content="Hello!")
        result = chat_hugging_face._to_chatml_format(msg)
        assert result == {"role": "assistant", "content": "Hello!"}
        assert "tool_calls" not in result


class TestToChatPromptWithTools:
    """Test _to_chat_prompt with tools parameter."""

    def test_tools_passed_to_apply_chat_template(self, chat_hugging_face: Any) -> None:
        messages = [HumanMessage(content="What's the weather?")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        chat_hugging_face.tokenizer.apply_chat_template.return_value = "prompt"

        chat_hugging_face._to_chat_prompt(messages, tools=tools)

        chat_hugging_face.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "What's the weather?"}],
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )

    def test_no_tools_omits_kwarg(self, chat_hugging_face: Any) -> None:
        messages = [HumanMessage(content="Hi")]
        chat_hugging_face.tokenizer.apply_chat_template.return_value = "prompt"

        chat_hugging_face._to_chat_prompt(messages)

        chat_hugging_face.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_tool_message_as_last_message(self, chat_hugging_face: Any) -> None:
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="get_weather", args={"city": "Paris"}, id="call_1"),
                ],
            ),
            ToolMessage(content="22C", tool_call_id="call_1"),
        ]
        chat_hugging_face.tokenizer.apply_chat_template.return_value = "prompt"

        result = chat_hugging_face._to_chat_prompt(messages)
        assert result == "prompt"


class TestParseToolCallsFromText:
    """Test _parse_tool_calls_from_text with various model output formats."""

    def test_tool_call_tags(self) -> None:
        text = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>"
        )
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"city": "Paris"}

    def test_multiple_tool_call_tags(self) -> None:
        text = (
            '<tool_call>{"name": "get_weather", "arguments": '
            '{"city": "Paris"}}</tool_call>\n'
            '<tool_call>{"name": "get_population", "arguments": '
            '{"city": "London"}}</tool_call>'
        )
        _content, tool_calls = _parse_tool_calls_from_text(text)
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[1]["name"] == "get_population"

    def test_raw_json_object(self) -> None:
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"city": "NYC"}

    def test_json_array(self) -> None:
        text = (
            '[{"name": "get_weather", "arguments": {"city": "LA"}}, '
            '{"name": "get_population", "arguments": {"city": "LA"}}]'
        )
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == ""
        assert len(tool_calls) == 2

    def test_plain_text_no_tool_calls(self) -> None:
        text = "The weather in Paris is sunny and 22 degrees."
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == text
        assert tool_calls == []

    def test_arguments_as_string(self) -> None:
        text = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}'
        _content, tool_calls = _parse_tool_calls_from_text(text)
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"] == {"city": "Paris"}

    def test_parameters_key(self) -> None:
        text = '{"name": "get_weather", "parameters": {"city": "Paris"}}'
        _content, tool_calls = _parse_tool_calls_from_text(text)
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"] == {"city": "Paris"}

    def test_invalid_json_in_tags(self) -> None:
        text = "<tool_call>not valid json</tool_call>"
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == ""
        assert tool_calls == []

    def test_tool_call_with_content(self) -> None:
        text = (
            "I'll look that up for you.\n"
            '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        )
        content, tool_calls = _parse_tool_calls_from_text(text)
        assert content == "I'll look that up for you."
        assert len(tool_calls) == 1


class TestToChatResultWithToolCalls:
    """Test _to_chat_result with parse_tool_calls flag."""

    def test_parse_tool_calls_from_llm_result(self) -> None:
        llm_result = LLMResult(
            generations=[
                [
                    Generation(
                        text='{"name": "get_weather", "arguments": {"city": "Paris"}}'
                    )
                ]
            ]
        )
        result = ChatHuggingFace._to_chat_result(llm_result, parse_tool_calls=True)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    def test_no_parse_returns_plain_text(self) -> None:
        llm_result = LLMResult(
            generations=[
                [
                    Generation(
                        text='{"name": "get_weather", "arguments": {"city": "Paris"}}'
                    )
                ]
            ]
        )
        result = ChatHuggingFace._to_chat_result(llm_result, parse_tool_calls=False)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "get_weather" in msg.content
        assert msg.tool_calls == []


class TestGeneratePipelineWithTools:
    """Test that _generate passes tools through the pipeline path."""

    @patch(
        "langchain_huggingface.chat_models.huggingface"
        ".ChatHuggingFace._resolve_model_id"
    )
    def test_generate_with_tools_calls_to_chat_prompt(self, mock_resolve: Any) -> None:
        from langchain_huggingface.llms.huggingface_pipeline import (
            HuggingFacePipeline,
        )

        mock_pipeline_llm = Mock(spec=HuggingFacePipeline)
        mock_pipeline_llm.model_id = "test/model"
        mock_pipeline_llm._generate.return_value = LLMResult(
            generations=[
                [
                    Generation(
                        text='{"name": "get_weather", "arguments": {"city": "Paris"}}'
                    )
                ]
            ]
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        chat = ChatHuggingFace(llm=mock_pipeline_llm, tokenizer=mock_tokenizer)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        messages: list[BaseMessage] = [HumanMessage(content="What's the weather?")]
        result = chat._generate(messages, tools=tools)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "What's the weather?"}],
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
