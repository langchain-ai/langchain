import json

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)

from langchain_xai import ChatXAI

MODEL_NAME = "grok-4"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatXAI(model=MODEL_NAME)


def test_xai_model_param() -> None:
    llm = ChatXAI(model="foo")
    assert llm.model_name == "foo"
    llm = ChatXAI(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    ls_params = llm._get_ls_params()
    assert ls_params.get("ls_provider") == "xai"


def test_chat_xai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatXAI(
            model=MODEL_NAME,
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_chat_xai_extra_kwargs() -> None:
    """Test extra kwargs to chat xai."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatXAI(model=MODEL_NAME, foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatXAI(model=MODEL_NAME, foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatXAI(model=MODEL_NAME, foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]


def test_function_dict_to_message_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_human_with_name() -> None:
    message = {"role": "user", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai_with_name() -> None:
    message = {"role": "assistant", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_server_tools_param() -> None:
    """Test server_tools parameter handling."""
    # Test with single server tool
    llm = ChatXAI(model=MODEL_NAME, server_tools=[{"type": "web_search"}])
    assert llm.server_tools == [{"type": "web_search"}]

    # Test with multiple server tools
    llm = ChatXAI(
        model=MODEL_NAME,
        server_tools=[
            {"type": "web_search"},
            {"type": "x_search"},
            {"type": "code_execution"},
        ],
    )
    assert llm.server_tools is not None
    assert len(llm.server_tools) == 3

    # Test that server_tools is added to default params
    params = llm._default_params
    assert "server_tools" in params
    assert params["server_tools"] == llm.server_tools


def test_server_tools_and_search_parameters_conflict() -> None:
    """Test that using both server_tools and search_parameters raises error."""
    with pytest.raises(
        ValueError,
        match="Cannot set both 'search_parameters' and 'server_tools'",
    ):
        ChatXAI(
            model=MODEL_NAME,
            server_tools=[{"type": "web_search"}],
            search_parameters={"mode": "auto"},
        )


def test_search_parameters_deprecation_warning() -> None:
    """Test that search_parameters shows deprecation warning."""
    with pytest.warns(DeprecationWarning, match="search_parameters.*deprecated"):
        ChatXAI(model=MODEL_NAME, search_parameters={"mode": "auto"})
