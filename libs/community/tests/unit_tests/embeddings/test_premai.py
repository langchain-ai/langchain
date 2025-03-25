"""Test EmbaasEmbeddings embeddings"""

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import PremAIEmbeddings


@pytest.mark.requires("premai")
def test_api_key_is_string() -> None:
    llm = PremAIEmbeddings(  # type: ignore[call-arg]
        premai_api_key="secret-api-key",  # type: ignore[arg-type]
        project_id=8,
        model="fake-model",  # type: ignore[arg-type]
    )
    assert isinstance(llm.premai_api_key, SecretStr)


@pytest.mark.requires("premai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = PremAIEmbeddings(  # type: ignore[call-arg]
        premai_api_key="secret-api-key",  # type: ignore[arg-type]
        project_id=8,
        model="fake-model",  # type: ignore[arg-type]
    )
    print(llm.premai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
