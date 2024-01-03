from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.llms.predibase import Predibase


def test_api_key_is_string() -> None:
    llm = Predibase(predibase_api_key="secret-api-key")
    assert isinstance(llm.predibase_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = Predibase(predibase_api_key="secret-api-key")
    print(llm.predibase_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
