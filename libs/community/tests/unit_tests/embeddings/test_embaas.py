"""Test EmbaasEmbeddings embeddings"""

from pydantic import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import EmbaasEmbeddings


def test_api_key_is_string() -> None:
    llm = EmbaasEmbeddings(embaas_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.embaas_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = EmbaasEmbeddings(embaas_api_key="secret-api-key")  # type: ignore[arg-type]
    print(llm.embaas_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
