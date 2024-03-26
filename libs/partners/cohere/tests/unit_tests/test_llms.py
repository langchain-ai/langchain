"""Test Cohere API wrapper."""
import typing

import pytest
from langchain_core.pydantic_v1 import SecretStr

from langchain_cohere.llms import BaseCohere, Cohere


def test_cohere_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cohere api key is a secret key."""
    # test initialization from init
    assert isinstance(BaseCohere(cohere_api_key="1").cohere_api_key, SecretStr)

    # test initialization from env variable
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    assert isinstance(BaseCohere().cohere_api_key, SecretStr)


@pytest.mark.parametrize(
    "cohere,expected",
    [
        pytest.param(Cohere(cohere_api_key="test"), {}, id="defaults"),
        pytest.param(
            Cohere(
                # the following are arbitrary testing values which shouldn't be used:
                cohere_api_key="test",
                model="foo",
                temperature=0.1,
                max_tokens=2,
                k=3,
                p=4,
                frequency_penalty=0.5,
                presence_penalty=0.6,
                truncate="START",
            ),
            {
                "model": "foo",
                "temperature": 0.1,
                "max_tokens": 2,
                "k": 3,
                "p": 4,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.6,
                "truncate": "START",
            },
            id="with values set",
        ),
    ],
)
def test_default_params(cohere: Cohere, expected: typing.Dict) -> None:
    actual = cohere._default_params
    assert expected == actual


# def test_saving_loading_llm(tmp_path: Path) -> None:
#     """Test saving/loading an Cohere LLM."""
#     llm = BaseCohere(max_tokens=10)
#     llm.save(file_path=tmp_path / "cohere.yaml")
#     loaded_llm = load_llm(tmp_path / "cohere.yaml")
#     assert_llm_equality(llm, loaded_llm)
