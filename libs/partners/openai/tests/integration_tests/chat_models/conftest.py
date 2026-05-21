"""Shared fixtures for chat-model integration tests.

The `ChatOpenAICodex` integration tests run under VCR cassette playback in
CI (`make test_vcr`), but its `FileChatGPTOAuthTokenProvider` still tries
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

import pytest

from langchain_openai import chatgpt_oauth


@pytest.fixture(autouse=True)
def _fake_codex_oauth_token(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub `FileChatGPTOAuthTokenProvider` token reads for Codex VCR tests."""
    if "codex" not in request.module.__name__:
        return

    fake_token = chatgpt_oauth.ChatGPTToken(
        access_token="vcr-fake-access-token",  # noqa: S106
        refresh_token="vcr-fake-refresh-token",  # noqa: S106
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        account_id="vcr-fake-account-id",
    )

    def _get_token(
        self: chatgpt_oauth.FileChatGPTOAuthTokenProvider,
    ) -> chatgpt_oauth.ChatGPTToken:
        return fake_token

    async def _aget_token(
        self: chatgpt_oauth.FileChatGPTOAuthTokenProvider,
    ) -> chatgpt_oauth.ChatGPTToken:
        return fake_token

    def _get_access_token(
        self: chatgpt_oauth.FileChatGPTOAuthTokenProvider,
    ) -> str:
        return fake_token.access_token

    async def _aget_access_token(
        self: chatgpt_oauth.FileChatGPTOAuthTokenProvider,
    ) -> str:
        return fake_token.access_token

    monkeypatch.setattr(
        chatgpt_oauth.FileChatGPTOAuthTokenProvider, "get_token", _get_token
    )
    monkeypatch.setattr(
        chatgpt_oauth.FileChatGPTOAuthTokenProvider, "aget_token", _aget_token
    )
    monkeypatch.setattr(
        chatgpt_oauth.FileChatGPTOAuthTokenProvider,
        "get_access_token",
        _get_access_token,
    )
    monkeypatch.setattr(
        chatgpt_oauth.FileChatGPTOAuthTokenProvider,
        "aget_access_token",
        _aget_access_token,
    )
