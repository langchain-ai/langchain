import json
import os
from unittest.mock import patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.chat_models.pipeshift import ChatPipeshift


@pytest.mark.requires("openai")
def test_initialization() -> None:
    """Test chat model initialization."""
    with patch.dict(os.environ, {"PIPESHIFT_API_KEY": "dummy_key"}):
        ChatPipeshift()  # type: ignore[call-arg]


@pytest.mark.requires("openai")
def test_pipeshift_model_param() -> None:
    with patch.dict(os.environ, {"PIPESHIFT_API_KEY": "dummy_key"}):
        llm = ChatPipeshift(model="foo")  # type: ignore[call-arg]
        assert llm.model == "foo"
        llm = ChatPipeshift(model="foo")  # type: ignore[call-arg]
        assert llm.model == "foo"
        ls_params = llm._get_ls_params()
        assert ls_params["ls_provider"] == "pipeshift"


def test_function_dict_to_message_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def testconvert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert convert_message_to_dict(expected_output) == message


def testconvert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert convert_message_to_dict(expected_output) == message


def testconvert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert convert_message_to_dict(expected_output) == message


def testconvert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert convert_message_to_dict(expected_output) == message


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-8bfDvknQda3SQ",
        "object": "chat.completion",
        "created": 1288983000,
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "name": "Ironyman",
                },
                "finish_reason": "stop",
            }
        ],
    }
