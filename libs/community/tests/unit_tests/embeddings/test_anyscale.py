"""Test AnyscaleEmbeddings embeddings"""

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import AnyscaleEmbeddings


def test_api_key_is_string() -> None:
    llm = AnyscaleEmbeddings(anyscale_api_key="secret-api-key")
    assert isinstance(llm.anyscale_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = AnyscaleEmbeddings(anyscale_api_key="secret-api-key")
    print(llm.anyscale_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
