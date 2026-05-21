"""Unit tests for `ChatOpenAICodex`."""
# ruff: noqa: S106, S107

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_openai import ChatOpenAICodex
from langchain_openai.chat_models.codex import (
    ACCOUNT_ID_HEADER,
    CHATGPT_CODEX_BASE_URL,
    DEFAULT_INSTRUCTIONS,
    ORIGINATOR_ENV_VAR,
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

    async def aget_access_token(self) -> str:
        token = await self.aget_token()
        return token.access_token


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
    assert model.store is False
    assert model.streaming is True


def test_uses_callable_api_key_from_token_provider() -> None:
    """The SDK-facing `api_key` must resolve to the provider's current token."""
    provider = FakeTokenProvider(access_token="abc")
    model = _build_model(token_provider=provider)
    secret = model.openai_api_key
    assert secret is not None
    if callable(secret):
        assert secret() == "abc"
    else:
        # `ChatOpenAI.validate_environment` wraps callables in `SecretStr`.
        assert secret.get_secret_value() == "abc"
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


def test_constructor_originator_overrides_default() -> None:
    """An explicit `originator=` constructor value replaces the package default."""
    model = _build_model(originator="my-app/1.2")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "my-app/1.2"


def test_env_var_overrides_default_originator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`LANGCHAIN_CODEX_ORIGINATOR` sets the default when no constructor value."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "env-app/2.0")
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "env-app/2.0"


def test_constructor_originator_wins_over_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit constructor value beats the env var (resolution order)."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "env-app/2.0")
    model = _build_model(originator="ctor-app/9.9")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "ctor-app/9.9"


def test_empty_env_var_falls_back_to_package_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty `LANGCHAIN_CODEX_ORIGINATOR` is treated as unset."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "")
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_caller_extra_headers_override_originator_field() -> None:
    """A per-call `extra_headers` originator beats the model's field value."""
    model = _build_model(originator="ctor-app")
    payload = model._get_request_payload(
        [HumanMessage("hi")], extra_headers={ORIGINATOR_HEADER: "call-app"}
    )
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "call-app"


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


def test_conflicting_use_responses_api_raises() -> None:
    with pytest.raises(ValueError, match="use_responses_api"):
        ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            use_responses_api=False,
        )


def test_conflicting_output_version_raises() -> None:
    with pytest.raises(ValueError, match="output_version"):
        ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            output_version="v0",
        )


def test_conflicting_store_raises() -> None:
    with pytest.raises(ValueError, match="store"):
        ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            store=True,
        )


def test_conflicting_streaming_raises() -> None:
    with pytest.raises(ValueError, match="streaming"):
        ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            streaming=False,
        )


def test_request_payload_sends_store_false_and_stream_true() -> None:
    """The Codex backend 400s with `store=True` or non-streaming requests."""
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["store"] is False
    assert payload["stream"] is True


def test_request_payload_sets_default_instructions() -> None:
    """The Codex backend 400s without `instructions`; default must be injected."""
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["instructions"] == DEFAULT_INSTRUCTIONS


def test_request_payload_respects_constructor_instructions() -> None:
    model = _build_model(instructions="custom system prompt")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["instructions"] == "custom system prompt"


def test_request_payload_respects_per_call_instructions_override() -> None:
    """An `instructions` kwarg at invoke time wins over the model default."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [HumanMessage("hi")], instructions="call-level"
    )
    assert payload["instructions"] == "call-level"


def test_request_payload_preserves_explicit_empty_instructions() -> None:
    """An explicit empty `instructions=""` must not be overwritten silently.

    The backend will reject it, but silently replacing it with the default
    would hide the caller's bug.
    """
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload([HumanMessage("hi")], instructions="")
    assert payload["instructions"] == ""


def test_system_message_does_not_populate_top_level_instructions() -> None:
    """A `SystemMessage` in `input` doesn't fill the top-level `instructions`.

    `SystemMessage` becomes an entry in the `input` list, leaving the
    separate top-level `instructions` field at the model's configured value.
    """
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [SystemMessage("from-system-message"), HumanMessage("hi")]
    )
    assert payload["instructions"] == "model-level"


def test_caller_headers_win_over_codex_defaults() -> None:
    """Caller-supplied `extra_headers` must override the Codex injections."""
    provider = FakeTokenProvider(account_id="acct-1")
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload(
        [HumanMessage("hi")],
        extra_headers={ORIGINATOR_HEADER: "custom-app"},
    )
    headers = payload["extra_headers"]
    assert headers[ORIGINATOR_HEADER] == "custom-app"
    # The auto-injected account ID still rides along when not overridden.
    assert headers[ACCOUNT_ID_HEADER] == "acct-1"


async def test_agenerate_primes_async_token_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_agenerate` must call `aget_token` so refresh doesn't block the loop."""
    provider = FakeTokenProvider()
    model = _build_model(token_provider=provider)

    async def _fake_super_agenerate(*_a: Any, **_k: Any) -> Any:
        return "sentinel"

    monkeypatch.setattr(
        "langchain_openai.chat_models.base.ChatOpenAI._agenerate",
        _fake_super_agenerate,
    )
    before = provider.async_calls
    result = await model._agenerate([HumanMessage("hi")])
    assert result == "sentinel"
    assert provider.async_calls == before + 1


def test_callable_api_key_returns_provider_token() -> None:
    """The `api_key` callable wired into the SDK must yield the access token."""
    provider = FakeTokenProvider(access_token="abc-123")
    model = _build_model(token_provider=provider)
    # ChatOpenAI converts callable api_keys into a `SecretStr` wrapping
    # whatever the callable returns; resolving it should return the
    # provider's current access token.
    secret = model.openai_api_key
    assert secret is not None
    if callable(secret):
        assert secret() == "abc-123"
    else:
        assert secret.get_secret_value() == "abc-123"


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
