import json

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from pydantic import SecretStr

from langchain_xai import ChatXAI

MODEL_NAME = "grok-4"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatXAI(model=MODEL_NAME)


def test_profile() -> None:
    model = ChatXAI(model="grok-4")
    assert model.profile


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


def test_chat_xai_base_url_alias() -> None:
    llm = ChatXAI(
        model=MODEL_NAME,
        api_key=SecretStr("test-api-key"),
        base_url="http://example.test/v1",
    )
    assert llm.xai_api_base == "http://example.test/v1"
    assert llm.model_kwargs == {}


def test_chat_xai_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_BASE", "http://env.example.test/v1")

    llm = ChatXAI(
        model=MODEL_NAME,
        api_key=SecretStr("test-api-key"),
    )

    assert llm.xai_api_base == "http://env.example.test/v1"


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


def test_stream_usage_metadata() -> None:
    model = ChatXAI(model=MODEL_NAME)
    assert model.stream_usage is True

    model = ChatXAI(model=MODEL_NAME, stream_usage=False)
    assert model.stream_usage is False


def test_response_metadata_model_provider() -> None:
    """Test that model_provider is correctly set in response_metadata."""
    llm = ChatXAI(model=MODEL_NAME, api_key=SecretStr("test-api-key"))
    
    # Mocking a raw OpenAI-style response dictionary
    mock_response = {
        "id": "test-id",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "model": MODEL_NAME,
        "object": "chat.completion",
    }
    
    # We call the private method you saw earlier to see if it injects the metadata
    res = llm._create_chat_result(mock_response)
    
    # Assert that our logic worked
    assert res.generations[0].message.response_metadata["model_provider"] == "xai"

def test_response_metadata_model_provider_streaming() -> None:
    """Test that model_provider is correctly set in streaming response_metadata."""
    llm = ChatXAI(model=MODEL_NAME, api_key=SecretStr("test-api-key"))
    
    # Mocking a raw streaming chunk
    mock_chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }
        ],
    }
    
    # We call the internal chunk conversion method
    res = llm._convert_chunk_to_generation_chunk(mock_chunk, AIMessageChunk, None)
    
    # Assert that the metadata exists in the chunk
    assert res.message.response_metadata["model_provider"] == "xai"