"""Shared fixtures for `langchain-openai` unit tests."""

import pytest

# Set by CI (and by developers routing local traffic through the LangSmith
# gateway). `ChatOpenAI` resolves these at construction time and swaps in the
# gateway base URL and key, which flips `_uses_gateway` and diverts mocked
# clients down the `with_raw_response` path.
_GATEWAY_VARS = (
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_API_KEY",
)

_PROVIDER_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROXY",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate unit tests from ambient OpenAI and gateway configuration.

    Unit tests run offline against mocked clients, so any real credential or
    base URL in the environment can only change behavior for the worse. Tests
    that exercise environment handling override these with their own
    `monkeypatch` calls, which still take precedence.
    """
    for var in (*_GATEWAY_VARS, *_PROVIDER_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "foo")
