"""Test EdenAiEmbeddings embeddings"""

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import EdenAiEmbeddings


def test_api_key_is_string() -> None:
    llm = EdenAiEmbeddings(edenai_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.edenai_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = EdenAiEmbeddings(edenai_api_key="secret-api-key")  # type: ignore[arg-type]
    print(llm.edenai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
