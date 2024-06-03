"""Test CerebriumAI llm"""

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.cerebriumai import CerebriumAI


def test_api_key_is_secret_string() -> None:
    llm = CerebriumAI(cerebriumai_api_key="test-cerebriumai-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.cerebriumai_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    llm = CerebriumAI(cerebriumai_api_key="secret-api-key")  # type: ignore[arg-type]
    print(llm.cerebriumai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert repr(llm.cerebriumai_api_key) == "SecretStr('**********')"


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("CEREBRIUMAI_API_KEY", "secret-api-key")
    llm = CerebriumAI()
    print(llm.cerebriumai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert repr(llm.cerebriumai_api_key) == "SecretStr('**********')"
