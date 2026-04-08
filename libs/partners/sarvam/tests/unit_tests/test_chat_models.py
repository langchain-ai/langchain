"""Unit tests for ChatSarvam."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_sarvam.chat_models import ChatSarvam, _convert_message_to_dict


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


class TestConvertMessageToDict:
    def test_human_message(self) -> None:
        from langchain_core.messages import HumanMessage

        result = _convert_message_to_dict(HumanMessage(content="hello"))
        assert result == {"role": "user", "content": "hello"}

    def test_ai_message(self) -> None:
        from langchain_core.messages import AIMessage

        result = _convert_message_to_dict(AIMessage(content="world"))
        assert result == {"role": "assistant", "content": "world"}

    def test_system_message(self) -> None:
        from langchain_core.messages import SystemMessage

        result = _convert_message_to_dict(SystemMessage(content="be helpful"))
        assert result == {"role": "system", "content": "be helpful"}

    def test_chat_message(self) -> None:
        from langchain_core.messages import ChatMessage

        result = _convert_message_to_dict(ChatMessage(role="user", content="hi"))
        assert result == {"role": "user", "content": "hi"}

    def test_unsupported_message_raises(self) -> None:
        from langchain_core.messages import BaseMessage

        class UnknownMsg(BaseMessage):
            type: str = "unknown"

            @property
            def _msg_type(self) -> str:
                return "unknown"

        with pytest.raises(ValueError, match="Unsupported message type"):
            _convert_message_to_dict(UnknownMsg(content="x"))


# ---------------------------------------------------------------------------
# ChatSarvam initialisation
# ---------------------------------------------------------------------------


def _build_mock_client() -> MagicMock:
    """Return a MagicMock that looks enough like SarvamAI for tests."""
    mock = MagicMock()
    # Fake a successful non-streaming response
    completion = MagicMock()
    completion.model_dump.return_value = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "sarvam-m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "New Delhi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    mock.chat.completions.return_value = completion
    return mock


class TestChatSarvamInit:
    def test_default_model(self) -> None:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            model = ChatSarvam(api_key="test-key")
        assert model.model_name == "sarvam-m"

    def test_model_alias(self) -> None:
        """The ``model`` alias should set ``model_name``."""
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            m = ChatSarvam(model="sarvam-30b", api_key="test-key")
        assert m.model_name == "sarvam-30b"

    def test_missing_package_raises(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sarvamai":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="sarvamai"):
                ChatSarvam(api_key="key")

    def test_llm_type(self) -> None:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            m = ChatSarvam(api_key="test-key")
        assert m._llm_type == "sarvam-chat"

    def test_is_serializable(self) -> None:
        assert ChatSarvam.is_lc_serializable() is True

    def test_lc_secrets(self) -> None:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            m = ChatSarvam(api_key="test-key")
        assert "sarvam_api_key" in m.lc_secrets


# ---------------------------------------------------------------------------
# _default_params
# ---------------------------------------------------------------------------


class TestDefaultParams:
    def _make_model(self, **kwargs: Any) -> ChatSarvam:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            return ChatSarvam(api_key="key", **kwargs)

    def test_basic_defaults(self) -> None:
        m = self._make_model()
        p = m._default_params
        assert p["model"] == "sarvam-m"
        assert p["temperature"] == 0.7
        assert "max_tokens" not in p

    def test_max_tokens_included_when_set(self) -> None:
        m = self._make_model(max_tokens=512)
        assert m._default_params["max_tokens"] == 512

    def test_reasoning_effort_included_when_set(self) -> None:
        m = self._make_model(reasoning_effort="high")
        assert m._default_params["reasoning_effort"] == "high"

    def test_stop_included_when_set(self) -> None:
        m = self._make_model(stop_sequences=["END"])
        assert m._default_params["stop"] == ["END"]


# ---------------------------------------------------------------------------
# _generate (mocked client)
# ---------------------------------------------------------------------------


class TestGenerate:
    def _make_model(self, **kwargs: Any) -> ChatSarvam:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            m = ChatSarvam(api_key="key", **kwargs)
        return m

    def test_generate_calls_client(self) -> None:
        from langchain_core.messages import HumanMessage

        m = self._make_model()
        mock_client = _build_mock_client()
        m.client = mock_client

        result = m._generate([HumanMessage(content="Capital of India?")])

        mock_client.chat.completions.assert_called_once()
        call_kwargs = mock_client.chat.completions.call_args
        assert call_kwargs.kwargs["messages"][0] == {
            "role": "user",
            "content": "Capital of India?",
        }
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "New Delhi"

    def test_usage_metadata_attached(self) -> None:
        from langchain_core.messages import HumanMessage

        m = self._make_model()
        m.client = _build_mock_client()

        result = m._generate([HumanMessage(content="Hello")])
        ai_msg = result.generations[0].message
        assert ai_msg.usage_metadata is not None
        assert ai_msg.usage_metadata["input_tokens"] == 10
        assert ai_msg.usage_metadata["output_tokens"] == 5

    def test_stop_sequences_forwarded(self) -> None:
        from langchain_core.messages import HumanMessage

        m = self._make_model()
        m.client = _build_mock_client()

        m._generate([HumanMessage(content="Hi")], stop=["END"])

        call_kwargs = m.client.chat.completions.call_args.kwargs
        assert call_kwargs["stop"] == ["END"]


# ---------------------------------------------------------------------------
# _stream (mocked client)
# ---------------------------------------------------------------------------


class TestStream:
    def _make_model(self, **kwargs: Any) -> ChatSarvam:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            return ChatSarvam(api_key="key", **kwargs)

    def _fake_stream_response(self) -> list[dict]:
        return [
            {
                "choices": [
                    {"delta": {"role": "assistant", "content": "New "}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"delta": {"content": "Delhi"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"delta": {}, "finish_reason": "stop"}
                ]
            },
        ]

    def test_stream_yields_chunks(self) -> None:
        from langchain_core.messages import HumanMessage

        m = self._make_model()
        m.client = MagicMock()
        m.client.chat.completions.return_value = self._fake_stream_response()

        chunks = list(m._stream([HumanMessage(content="Capital?")]))
        text = "".join(c.message.content for c in chunks)
        assert "New Delhi" in text

    def test_stream_sends_stream_true(self) -> None:
        from langchain_core.messages import HumanMessage

        m = self._make_model()
        m.client = MagicMock()
        m.client.chat.completions.return_value = self._fake_stream_response()

        list(m._stream([HumanMessage(content="Hi")]))
        assert m.client.chat.completions.call_args.kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncGenerate:
    def _make_model(self, **kwargs: Any) -> ChatSarvam:
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            return ChatSarvam(api_key="key", **kwargs)

    @pytest.mark.asyncio
    async def test_agenerate_calls_async_client(self) -> None:
        from unittest.mock import AsyncMock

        from langchain_core.messages import HumanMessage

        m = self._make_model()
        mock_async = MagicMock()
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-xyz",
            "model": "sarvam-m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mumbai"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        mock_async.chat.completions = AsyncMock(return_value=completion)
        m.async_client = mock_async

        result = await m._agenerate([HumanMessage(content="Largest city in India?")])
        assert result.generations[0].message.content == "Mumbai"
