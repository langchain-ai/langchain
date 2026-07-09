"""Test Fireworks LLM."""

from typing import cast
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch
from typing_extensions import Self

from langchain_fireworks import Fireworks


def test_fireworks_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.fireworks_api_key, SecretStr)

    # Test api_key alias
    llm = Fireworks(
        api_key="secret-api-key",  # type: ignore[arg-type]
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.fireworks_api_key, SecretStr)


def test_fireworks_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-api-key")
    llm = Fireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.fireworks_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_fireworks_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.fireworks_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_fireworks_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert cast(SecretStr, llm.fireworks_api_key).get_secret_value() == "secret-api-key"


def test_fireworks_model_params() -> None:
    # Test standard tracing params
    llm = Fireworks(model="foo", api_key="secret-api-key")  # type: ignore[arg-type]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "fireworks",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
    }

    llm = Fireworks(
        model="foo",
        api_key="secret-api-key",  # type: ignore[arg-type]
        max_tokens=10,
        temperature=0.1,
    )

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "fireworks",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 10,
        "ls_temperature": 0.1,
    }


class _FakeAsyncResponse:
    """Minimal stand-in for an `aiohttp.ClientResponse` used as an async CM."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body

    async def json(self) -> dict:
        return {}


class _FakeAsyncSession:
    """Minimal stand-in for an `aiohttp.ClientSession` used as an async CM."""

    def __init__(self, response: _FakeAsyncResponse) -> None:
        self._response = response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def post(self, *args: object, **kwargs: object) -> _FakeAsyncResponse:
        return self._response


async def test_fireworks_acall_error_surfaces_response_body() -> None:
    """Async error paths should include the API response body, not a coroutine repr.

    On an `aiohttp` response, `.text` is a coroutine method, not a property (as it is
    on a `requests` response), so `_acall` must `await` it. Regression test: the raised
    error must contain the real API message, not `<bound method ...>`.
    """
    response = _FakeAsyncResponse(status=400, body="model does not exist")
    llm = Fireworks(model="foo", api_key="secret-api-key")  # type: ignore[arg-type]
    with (
        patch(
            "langchain_fireworks.llms.ClientSession",
            return_value=_FakeAsyncSession(response),
        ),
        pytest.raises(ValueError, match="model does not exist") as exc_info,
    ):
        await llm._acall("hi")
    assert "bound method" not in str(exc_info.value)
