from typing import Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult
from transformers import AutoTokenizer

from langchain_huggingface.chat_models import (  # type: ignore[import]
    TGI_MESSAGE,
    ChatHuggingFace,
    _convert_message_to_chat_message,
    _convert_TGI_message_to_LC_message,
)
from langchain_huggingface.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
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
            TGI_MESSAGE(role="assistant", content="Hello", tool_calls=None),
            AIMessage(content="Hello"),
        ),
        (
            TGI_MESSAGE(role="assistant", content=None, tool_calls=None),
            AIMessage(content=""),
        ),
        (
            TGI_MESSAGE(
                role="assistant",
                content=None,
                tool_calls=[{"function": {"parameters": "'function string'"}}],
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [{"function": {"arguments": '"function string"'}}]
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
def mock_llm():
    llm = Mock(spec=HuggingFaceTextGenInference)
    llm.inference_server_url = "test tgi url"
    return llm


@pytest.fixture
def mock_tokenizer():
    with patch.object(
        AutoTokenizer, "from_pretrained", return_value=MagicMock()
    ) as mock:
        yield mock


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)
def chat_hugging_face(mock_resolve_id, mock_llm, mock_tokenizer):
    chat_hf = ChatHuggingFace(llm=mock_llm, tokenizer=mock_tokenizer)
    return chat_hf


def test_create_chat_result(chat_hugging_face):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=TGI_MESSAGE(
                role="assistant", content="test message", tool_calls=None
            ),
            finish_reason="test finish reason",
        )
    ]
    mock_response.usage = {"tokens": 420}

    result = chat_hugging_face._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"
    )
    assert result.llm_output["token_usage"]["tokens"] == 420
    assert result.llm_output["model"] == chat_hugging_face.llm.inference_server_url
