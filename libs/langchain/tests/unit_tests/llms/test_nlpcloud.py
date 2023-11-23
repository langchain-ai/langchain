"""Test NLPCloud"""

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.nlpcloud import NLPCloud


def test_api_key_is_secret_string() -> None:
    llm = NLPCloud(nlpcloud_api_key="secret-api-key")
    assert isinstance(llm.nlpcloud_api_key, SecretStr)
    assert llm.nlpcloud_api_key.get_secret_value() == "secret-api-key"


def test_api_key_masked_when_passed_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("NLPCLOUD_API_KEY", "secret-api-key")
    llm = NLPCloud()
    assert llm.nlpcloud_api_key is None


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    llm = NLPCloud(nlpcloud_api_key="secret-api-key")
    print(str(llm.nlpcloud_api_key), end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"
