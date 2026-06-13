"""Shared fixtures for chat-model integration tests.

The `_ChatOpenAICodex` integration tests run under VCR cassette playback in
CI (`make test_vcr`), but its `_FileChatGPTOAuthTokenProvider` still tries
to read `~/.langchain/chatgpt-auth.json` from disk on every request to
build the `Authorization` header. CI has no such file, so every Codex
test would fail with `FileNotFoundError` before VCR ever replays the
cassette.

This fixture monkey-patches the on-disk token methods with an in-memory
fake token whenever a Codex test module is running, so cassette replay
works without a live OAuth login. The patch is scoped by module name and
leaves non-Codex tests in the directory untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from langchain_openai import chatgpt_oauth


@pytest.fixture(autouse=True)
def _clear_openai_base_url_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strip ambient OpenAI endpoint env vars for VCR tests only.

    `ChatOpenAI` only default-enables `stream_usage` when no custom base URL is
    configured (see the `OPENAI_BASE_URL` / `OPENAI_API_BASE` handling at
    construction time). A developer whose shell exports one of these vars (e.g.
    pointing at a gateway) would otherwise see `stream_usage` silently dropped,
    so the request body omits `stream_options` and no longer matches the
    recorded cassette's `json_body` — VCR then attempts a live call and fails
    with `APIConnectionError`.

    Scoped to `@pytest.mark.vcr` tests: cassettes are recorded against the
    canonical `api.openai.com` host, so cassette playback (and re-recording)
    must ignore an ambient gateway endpoint. Live integration tests (no
    cassette) are left untouched so a developer can still route them through a
    gateway via these env vars.
    """
    if request.node.get_closest_marker("vcr") is None:
        return
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)


def _vcr_record_mode(config: pytest.Config) -> str | None:
    """Return pytest-recording's configured record mode, if available."""
    for option in ("record_mode", "--record-mode"):
        try:
            value: Any = config.getoption(option, default=None)
        except ValueError:
            continue
        if value is not None:
            return str(value)
    return None


@pytest.fixture(autouse=True)
def _fake_codex_oauth_token(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub `_FileChatGPTOAuthTokenProvider` token reads for Codex VCR tests."""
    if "codex" not in request.module.__name__:
        return
    if _vcr_record_mode(request.config) != "none":
        return

    fake_token = chatgpt_oauth._ChatGPTToken(
        access_token="vcr-fake-access-token",  # noqa: S106
        refresh_token="vcr-fake-refresh-token",  # noqa: S106
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        account_id="vcr-fake-account-id",
    )

    def _get_token(
        self: chatgpt_oauth._FileChatGPTOAuthTokenProvider,
    ) -> chatgpt_oauth._ChatGPTToken:
        return fake_token

    async def _aget_token(
        self: chatgpt_oauth._FileChatGPTOAuthTokenProvider,
    ) -> chatgpt_oauth._ChatGPTToken:
        return fake_token

    def _get_access_token(
        self: chatgpt_oauth._FileChatGPTOAuthTokenProvider,
    ) -> str:
        return fake_token.access_token

    async def _aget_access_token(
        self: chatgpt_oauth._FileChatGPTOAuthTokenProvider,
    ) -> str:
        return fake_token.access_token

    monkeypatch.setattr(
        chatgpt_oauth._FileChatGPTOAuthTokenProvider, "get_token", _get_token
    )
    monkeypatch.setattr(
        chatgpt_oauth._FileChatGPTOAuthTokenProvider, "aget_token", _aget_token
    )
    monkeypatch.setattr(
        chatgpt_oauth._FileChatGPTOAuthTokenProvider,
        "get_access_token",
        _get_access_token,
    )
    monkeypatch.setattr(
        chatgpt_oauth._FileChatGPTOAuthTokenProvider,
        "aget_access_token",
        _aget_access_token,
    )
