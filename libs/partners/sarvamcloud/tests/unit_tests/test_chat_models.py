"""Unit tests for ChatSarvam."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_sarvamcloud.chat_models import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    _lc_tool_call_to_sarvam_tool_call,
)


@pytest.fixture()
def mock_sarvam_client() -> MagicMock:
    """Return a mock SarvamAI client."""
    client = MagicMock()
    client.chat.completions.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "sarvam-105b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return client


@pytest.fixture()
def chat_sarvam(mock_sarvam_client: MagicMock) -> Any:
    """Return a ChatSarvam instance with mocked client."""
    from langchain_sarvamcloud.chat_models import ChatSarvam

    with patch("langchain_sarvamcloud.chat_models.ChatSarvam.validate_environment"):
        model = ChatSarvam.__new__(ChatSarvam)
        model.__dict__.update(
            {
                "model_name": "sarvam-105b",
                "temperature": 0.2,
                "top_p": 1.0,
                "max_tokens": None,
                "reasoning_effort": None,
                "streaming": False,
                "max_retries": 2,
                "api_subscription_key": None,
                "base_url": "https://api.sarvam.ai/v1",
                "model_kwargs": {},
                "client": mock_sarvam_client,
                "async_client": MagicMock(),
                "profile": None,
            }
        )
    return model


class TestMessageConversion:
    def test_human_message_to_dict(self) -> None:
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_system_message_to_dict(self) -> None:
        msg = SystemMessage(content="You are a helpful assistant.")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "system", "content": "You are a helpful assistant."}

    def test_ai_message_to_dict(self) -> None:
        msg = AIMessage(content="Hi there!")
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"

    def test_tool_message_to_dict(self) -> None:
        msg = ToolMessage(content='{"temp": 72}', tool_call_id="call_abc123")
        result = _convert_message_to_dict(msg)
        assert result["role"] == "tool"
        assert result["content"] == '{"temp": 72}'
        assert result["tool_call_id"] == "call_abc123"

    def test_ai_message_with_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "Delhi"}, "id": "call_1"}
            ],
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["content"] is None

    def test_unknown_message_raises(self) -> None:
        from langchain_core.messages import BaseMessage

        class UnknownMessage(BaseMessage):
            type: str = "unknown"

        msg = UnknownMessage(content="test")
        with pytest.raises(TypeError):
            _convert_message_to_dict(msg)


class TestDictToMessage:
    def test_user_dict_to_human_message(self) -> None:
        result = _convert_dict_to_message({"role": "user", "content": "Hello"})
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_assistant_dict_to_ai_message(self) -> None:
        result = _convert_dict_to_message(
            {"role": "assistant", "content": "I can help!"}
        )
        assert isinstance(result, AIMessage)
        assert result.content == "I can help!"

    def test_assistant_dict_with_tool_calls(self) -> None:
        raw = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Delhi"}',
                    },
                }
            ],
        }
        result = _convert_dict_to_message(raw)
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_system_dict_to_system_message(self) -> None:
        result = _convert_dict_to_message({"role": "system", "content": "Be helpful"})
        assert isinstance(result, SystemMessage)

    def test_tool_dict_to_tool_message(self) -> None:
        result = _convert_dict_to_message(
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}
        )
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_1"

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValueError, match="Got unknown role"):
            _convert_dict_to_message({"role": "unknown", "content": "test"})


class TestToolCallConversion:
    def test_lc_tool_call_to_sarvam(self) -> None:
        tc = {"name": "search", "args": {"query": "weather"}, "id": "call_xyz"}
        result = _lc_tool_call_to_sarvam_tool_call(tc)  # type: ignore[arg-type]
        assert result["type"] == "function"
        assert result["id"] == "call_xyz"
        assert result["function"]["name"] == "search"
        assert '"query"' in result["function"]["arguments"]


class TestChatSarvamDefaults:
    def test_default_temperature_is_0_2(self) -> None:
        """Sarvam's default temperature is 0.2, not 0.7."""
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key")
        assert model.temperature == 0.2

    def test_default_model_is_sarvam_105b(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key")
        assert model.model_name == "sarvam-105b"

    def test_reasoning_effort_default_is_none(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key")
        assert model.reasoning_effort is None

    def test_reasoning_effort_accepted_values(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        for effort in ("low", "medium", "high"):
            with (
                patch("sarvamai.SarvamAI"),
                patch("sarvamai.AsyncSarvamAI"),
            ):
                model = ChatSarvam(
                    api_subscription_key="test-key", reasoning_effort=effort
                )
            assert model.reasoning_effort == effort

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        monkeypatch.setenv("SARVAM_API_KEY", "env-test-key")
        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam()
        assert model.api_subscription_key is not None
        assert model.api_subscription_key.get_secret_value() == "env-test-key"

    def test_llm_type(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key")
        assert model._llm_type == "sarvam-chat"


class TestDefaultParams:
    def test_reasoning_effort_in_params_when_set(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(
                api_subscription_key="test-key", reasoning_effort="high"
            )
        params = model._default_params
        assert params["reasoning_effort"] == "high"

    def test_max_tokens_included_when_set(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key", max_tokens=512)
        params = model._default_params
        assert params["max_tokens"] == 512

    def test_max_tokens_not_in_params_when_none(self) -> None:
        from langchain_sarvamcloud.chat_models import ChatSarvam

        with (
            patch("sarvamai.SarvamAI"),
            patch("sarvamai.AsyncSarvamAI"),
        ):
            model = ChatSarvam(api_subscription_key="test-key")
        params = model._default_params
        assert "max_tokens" not in params


class TestModelProfiles:
    def test_sarvam_105b_context_window(self) -> None:
        from langchain_sarvamcloud.data._profiles import _PROFILES

        assert _PROFILES["sarvam-105b"]["max_input_tokens"] == 128000

    def test_sarvam_30b_context_window(self) -> None:
        from langchain_sarvamcloud.data._profiles import _PROFILES

        assert _PROFILES["sarvam-30b"]["max_input_tokens"] == 32000

    def test_sarvam_m_no_tool_use(self) -> None:
        from langchain_sarvamcloud.data._profiles import _PROFILES

        assert _PROFILES["sarvam-m"]["tool_use"] is False

    def test_sarvam_105b_tool_use(self) -> None:
        from langchain_sarvamcloud.data._profiles import _PROFILES

        assert _PROFILES["sarvam-105b"]["tool_use"] is True

    def test_all_five_models_present(self) -> None:
        from langchain_sarvamcloud.data._profiles import _PROFILES

        expected = {
            "sarvam-m",
            "sarvam-30b",
            "sarvam-30b-16k",
            "sarvam-105b",
            "sarvam-105b-32k",
        }
        assert expected.issubset(set(_PROFILES.keys()))
