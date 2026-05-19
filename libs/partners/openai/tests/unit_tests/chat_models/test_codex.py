"""Unit tests for `ChatOpenAICodex`."""
# ruff: noqa: S106, S107

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAICodex
from langchain_openai.chat_models.codex import (
    ACCOUNT_ID_HEADER,
    CHATGPT_CODEX_BASE_URL,
    ORIGINATOR_HEADER,
    ORIGINATOR_VALUE,
    _SyncTokenCallable,
)
from langchain_openai.chatgpt_oauth import ChatGPTToken


class FakeTokenProvider:
    """Minimal `ChatGPTOAuthTokenProvider` for tests."""

    def __init__(
        self,
        access_token: str = "at-1",
        account_id: str | None = "acct-1",
    ) -> None:
        self.access_token = access_token
        self.account_id = account_id
        self.calls = 0
        self.async_calls = 0

    def get_token(self) -> ChatGPTToken:
        self.calls += 1
        return ChatGPTToken(
            access_token=self.access_token,
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            account_id=self.account_id,
        )

    async def aget_token(self) -> ChatGPTToken:
        self.async_calls += 1
        return self.get_token()

    def get_access_token(self) -> str:
        return self.get_token().access_token


def _build_model(**overrides: Any) -> ChatOpenAICodex:
    provider = overrides.pop("token_provider", None) or FakeTokenProvider()
    return ChatOpenAICodex(
        model=overrides.pop("model", "gpt-5.2-codex"),
        token_provider=provider,
        **overrides,
    )


def test_defaults_route_to_chatgpt_codex_backend() -> None:
    model = _build_model()
    assert model.openai_api_base == CHATGPT_CODEX_BASE_URL
    assert model.use_responses_api is True
    assert model.output_version == "responses/v1"


def test_uses_callable_api_key_from_token_provider() -> None:
    provider = FakeTokenProvider(access_token="abc")
    model = _build_model(token_provider=provider)
    # OpenAI SDK stores the callable internally; the SecretStr wrapping in
    # `validate_environment` resolves to the callable for sync requests.
    assert callable(model.openai_api_key) or hasattr(
        model.openai_api_key, "get_secret_value"
    )
    callable_ = _SyncTokenCallable(provider)
    assert callable_() == "abc"
    assert provider.calls >= 1


def test_explicit_base_url_override_is_respected() -> None:
    custom = "https://custom.example.com/codex"
    model = _build_model(base_url=custom)
    assert model.openai_api_base == custom


def test_request_payload_injects_account_id_and_originator_headers() -> None:
    provider = FakeTokenProvider(account_id="acct-42")
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload([HumanMessage("hi")])
    headers = payload["extra_headers"]
    assert headers[ACCOUNT_ID_HEADER] == "acct-42"
    assert headers[ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_request_payload_omits_account_id_when_unknown() -> None:
    provider = FakeTokenProvider(account_id=None)
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload([HumanMessage("hi")])
    headers = payload["extra_headers"]
    assert ACCOUNT_ID_HEADER not in headers
    assert headers[ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_request_payload_can_disable_originator_header() -> None:
    provider = FakeTokenProvider(account_id=None)
    model = _build_model(token_provider=provider, include_originator_header=False)
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "extra_headers" not in payload


def test_request_payload_pulls_fresh_account_id_each_call() -> None:
    provider = FakeTokenProvider()
    model = _build_model(token_provider=provider)
    before = provider.calls
    model._get_request_payload([HumanMessage("hi")])
    model._get_request_payload([HumanMessage("hi again")])
    assert provider.calls >= before + 2


def test_invalid_token_provider_rejected() -> None:
    with pytest.raises(TypeError):
        ChatOpenAICodex(model="gpt-5.2-codex", token_provider="not-a-provider")


def test_ls_params_uses_codex_provider_tag() -> None:
    model = _build_model()
    params = model._get_ls_params()
    assert params["ls_provider"] == "openai-codex"


def test_is_not_serializable_due_to_live_token_provider() -> None:
    assert ChatOpenAICodex.is_lc_serializable() is False


def test_sync_token_callable_delegates() -> None:
    provider = FakeTokenProvider(access_token="zzz")
    callable_ = _SyncTokenCallable(provider)
    assert callable_() == "zzz"
