"""Test LLMRailsEmbeddings embeddings"""

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import LLMRailsEmbeddings


def test_api_key_is_string() -> None:
    llm = LLMRailsEmbeddings(api_key="secret-api-key")
    assert isinstance(llm.api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = LLMRailsEmbeddings(api_key="secret-api-key")
    print(llm.api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
