from typing import Any, Dict, List  # type: ignore[import-not-found]
from unittest.mock import MagicMock, Mock, patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool

from langchain_huggingface.chat_models import (  # type: ignore[import]
    TGI_MESSAGE,
    ChatHuggingFace,
    _convert_message_to_chat_message,
    _convert_TGI_message_to_LC_message,
)
from langchain_huggingface.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="Hello"),
            dict(role="system", content="Hello"),
        ),
        (
            HumanMessage(content="Hello"),
            dict(role="user", content="Hello"),
        ),
        (
            AIMessage(content="Hello"),
            dict(role="assistant", content="Hello", tool_calls=None),
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            dict(role="assistant", content="Hello"),
        ),
    ],
)
def test_convert_message_to_chat_message(
    message: BaseMessage, expected: Dict[str, str]
) -> None:
    result = _convert_message_to_chat_message(message)
    assert result == expected


@pytest.mark.parametrize(
    ("tgi_message", "expected"),
    [
        (
            TGI_MESSAGE(role="assistant", content="Hello", tool_calls=[]),
            AIMessage(content="Hello"),
        ),
        (
            TGI_MESSAGE(role="assistant", content="", tool_calls=[]),
            AIMessage(content=""),
        ),
        (
            TGI_MESSAGE(
                role="assistant",
                content="",
                tool_calls=[{"function": {"arguments": "function string"}}],
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [{"function": {"arguments": '"function string"'}}]
                },
            ),
        ),
        (
            TGI_MESSAGE(
                role="assistant",
                content="",
                tool_calls=[
                    {"function": {"arguments": {"answer": "function's string"}}}
                ],
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {"function": {"arguments": '{"answer": "function\'s string"}'}}
                    ]
                },
            ),
        ),
    ],
)
def test_convert_TGI_message_to_LC_message(
    tgi_message: TGI_MESSAGE, expected: BaseMessage
) -> None:
    result = _convert_TGI_message_to_LC_message(tgi_message)
    assert result == expected


@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.inference_server_url = "test endpoint url"
    return llm


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)
def chat_hugging_face(mock_resolve_id: Any, mock_llm: Any) -> ChatHuggingFace:
    chat_hf = ChatHuggingFace(llm=mock_llm, tokenizer=MagicMock())
    return chat_hf


def test_create_chat_result(chat_hugging_face: Any) -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=TGI_MESSAGE(
                role="assistant", content="test message", tool_calls=[]
            ),
            finish_reason="test finish reason",
        )
    ]
    mock_response.usage = {"tokens": 420}

    result = chat_hugging_face._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"  # type: ignore[index]
    )
    assert result.llm_output["token_usage"]["tokens"] == 420  # type: ignore[index]
    assert result.llm_output["model"] == chat_hugging_face.llm.inference_server_url  # type: ignore[index]


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "At least one HumanMessage must be provided!"),
        (
            [HumanMessage(content="Hi"), AIMessage(content="Hello")],
            "Last message must be a HumanMessage!",
        ),
    ],
)
def test_to_chat_prompt_errors(
    chat_hugging_face: Any, messages: List[BaseMessage], expected_error: str
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
    chat_hugging_face: Any, message: BaseMessage, expected: Dict[str, str]
) -> None:
    result = chat_hugging_face._to_chatml_format(message)
    assert result == expected


def test_to_chatml_format_with_invalid_type(chat_hugging_face: Any) -> None:
    message = "Invalid message type"
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chatml_format(message)
    assert "Unknown message type:" in str(e.value)


def tool_mock() -> Dict:
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
    tools: Dict[str, str],
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
    with patch(
        "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
        side_effect=lambda x: x,
    ), patch("langchain_core.runnables.base.Runnable.bind") as mock_super_bind:
        chat_hugging_face.bind_tools(tools, tool_choice="auto")
        mock_super_bind.assert_called_once()
        _, kwargs = mock_super_bind.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"
