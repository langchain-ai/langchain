"""Tests for the LangSmith model gateway chat model."""

import openai
import pytest
from langchain_core.load import dumps, loads
from pydantic import SecretStr

from langchain_openai import ChatLangSmithGateway


def _secret_value(model: ChatLangSmithGateway) -> str:
    assert isinstance(model.openai_api_key, SecretStr)
    return model.openai_api_key.get_secret_value()


def test_gateway_key_priority_and_serialization_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-key")

    model = ChatLangSmithGateway(model="custom/model")
    serialized = dumps(model)
    restored = loads(serialized, allowed_objects="all", secrets_from_env=True)

    assert model.openai_api_base == "https://gateway.smith.langchain.com/v1"
    assert model.use_responses_api is True
    assert model._uses_gateway is True
    assert model.lc_id() == [
        "langchain",
        "chat_models",
        "openai",
        "ChatLangSmithGateway",
    ]
    assert '"id": ["LANGSMITH_GATEWAY_API_KEY"]' in serialized
    assert isinstance(restored, ChatLangSmithGateway)
    assert _secret_value(restored) == "gateway-key"


def test_serialization_falls_back_to_langsmith_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-key")
    model = ChatLangSmithGateway(model="custom/model")

    restored = loads(
        dumps(model),
        allowed_objects="all",
        secrets_from_env=True,
    )

    assert isinstance(restored, ChatLangSmithGateway)
    assert _secret_value(restored) == "langsmith-key"


def test_explicit_url_and_responses_api_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")

    model = ChatLangSmithGateway(
        model="custom/model",
        base_url="https://gateway.example.com/v1",
        use_responses_api=False,
    )

    assert model.openai_api_base == "https://gateway.example.com/v1"
    assert model.use_responses_api is True


def test_missing_langsmith_credentials_ignores_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("LANGSMITH_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(openai.OpenAIError, match="Missing credentials"):
        ChatLangSmithGateway(model="custom/model")
