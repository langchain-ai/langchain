"""Test the Nebula model by Symbl.ai"""

from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.symblai_nebula import Nebula


def test_api_key_is_secret_string() -> None:
    llm = Nebula(nebula_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.nebula_api_key, SecretStr)
    assert llm.nebula_api_key.get_secret_value() == "secret-api-key"


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("NEBULA_API_KEY", "secret-api-key")
    llm = Nebula()
    print(llm.nebula_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    llm = Nebula(nebula_api_key="secret-api-key")  # type: ignore[arg-type]
    print(llm.nebula_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"
