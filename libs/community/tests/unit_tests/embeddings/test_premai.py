"""Test EmbaasEmbeddings embeddings"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.embeddings import PremAIEmbeddings


@pytest.mark.requires("premai")
def test_api_key_is_string() -> None:
    llm = PremAIEmbeddings(
        premai_api_key="secret-api-key", project_id=8, model="fake-model"
    )
    assert isinstance(llm.premai_api_key, SecretStr)


@pytest.mark.requires("premai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = PremAIEmbeddings(
        premai_api_key="secret-api-key", project_id=8, model="fake-model"
    )
    print(llm.premai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
