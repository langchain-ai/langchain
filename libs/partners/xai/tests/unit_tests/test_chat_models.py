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
from pydantic import SecretStr

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
    with pytest.warns(UserWarning, match="foo is not default parameter"):
        llm = ChatXAI(model=MODEL_NAME, foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    with pytest.warns(UserWarning, match="foo is not default parameter"):
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


@pytest.mark.parametrize(
    "model",
    [
        # Profiled reasoning models (`reasoning_output=True`).
        "grok-4.3",
        "grok-4.20-0309-reasoning",
        # Unprofiled families that the live API rejects `stop` on. `grok-4`
        # base and `grok-4-fast-non-reasoning` lack the substring "reasoning"
        # yet still reject `stop`; `grok-code-fast` is a separate family.
        "grok-3",
        "grok-3-mini",
        "grok-4",
        "grok-4-0709",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-code-fast-1",
    ],
)
def test_reasoning_model_payload_drops_stop(model: str) -> None:
    llm = ChatXAI(
        model=model,
        api_key=SecretStr("test-api-key"),
        stop_sequences=["END"],
    )

    payload = llm._get_request_payload("hello")

    assert "stop" not in payload


def test_non_reasoning_model_payload_keeps_stop() -> None:
    # `grok-4.20-0309-non-reasoning` is profiled with `reasoning_output=False`
    # and the live API accepts `stop` for it, even though its name contains
    # "non-reasoning" like the unprofiled `grok-4-fast-non-reasoning` that does
    # not. The profile must take precedence over the name-based fallback.
    llm = ChatXAI(
        model="grok-4.20-0309-non-reasoning",
        api_key=SecretStr("test-api-key"),
        stop_sequences=["END"],
    )

    payload = llm._get_request_payload("hello")

    assert payload["stop"] == ["END"]


def test_reasoning_effort_moved_to_extra_body() -> None:
    """`reasoning_effort` (inherited from `BaseChatOpenAI`) must reach xAI's
    API via `extra_body`, since xAI does not accept it as a top-level field.
    """
    llm = ChatXAI(
        model="grok-3-mini",
        api_key=SecretStr("test-api-key"),
        reasoning_effort="high",
    )

    payload = llm._get_request_payload("hello")

    assert "reasoning_effort" not in payload
    assert payload["extra_body"]["reasoning_effort"] == "high"


def test_reasoning_effort_as_call_time_kwarg() -> None:
    """`reasoning_effort` also works as a call-time keyword argument.

    This is the standard `reasoning_effort` param shared across chat model
    integrations, so it must work via `model.invoke(..., reasoning_effort=...)`
    without requiring it to be set on the model instance.
    """
    llm = ChatXAI(model="grok-3-mini", api_key=SecretStr("test-api-key"))

    payload = llm._get_request_payload("hello", reasoning_effort="low")

    assert "reasoning_effort" not in payload
    assert payload["extra_body"]["reasoning_effort"] == "low"


def test_reasoning_effort_preserves_existing_extra_body() -> None:
    """Moving `reasoning_effort` into `extra_body` must not drop sibling keys."""
    llm = ChatXAI(
        model="grok-3-mini",
        api_key=SecretStr("test-api-key"),
        reasoning_effort="high",
        extra_body={"some_other_field": "value"},
    )

    payload = llm._get_request_payload("hello")

    assert payload["extra_body"] == {
        "some_other_field": "value",
        "reasoning_effort": "high",
    }


def test_no_reasoning_effort_leaves_extra_body_untouched() -> None:
    llm = ChatXAI(
        model="grok-3-mini",
        api_key=SecretStr("test-api-key"),
        extra_body={"some_other_field": "value"},
    )

    payload = llm._get_request_payload("hello")

    assert payload["extra_body"] == {"some_other_field": "value"}
    assert "reasoning_effort" not in payload


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


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    llm = ChatXAI(model=MODEL_NAME)
    assert llm.metadata is not None
    versions = llm.metadata["lc_versions"]
    assert "langchain-core" in versions
    assert "langchain-xai" in versions
    assert "langchain-openai" in versions
