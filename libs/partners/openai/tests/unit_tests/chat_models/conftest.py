"""Shared fixtures for chat model unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _clear_openai_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient OpenAI endpoint env vars so unit tests stay deterministic.

    `ChatOpenAI` only default-enables `stream_usage` when no custom base URL is
    configured (see the `OPENAI_BASE_URL` / `OPENAI_API_BASE` handling at
    construction time). A developer whose shell exports one of these vars (e.g.
    pointing at a gateway) would otherwise see `stream_usage` silently dropped,
    breaking serialization snapshots that assume the default. Removing them here
    keeps results consistent with CI, where the vars are unset.
    """
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
