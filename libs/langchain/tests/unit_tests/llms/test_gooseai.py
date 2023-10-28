"""Test GooseAI"""

import pytest
from pytest import MonkeyPatch

from langchain.llms.gooseai import GooseAI
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("openai")
def test_api_key_is_secret_string() -> None:
    llm = GooseAI(gooseai_api_key="secret-api-key")
    assert isinstance(llm.gooseai_api_key, SecretStr)
    assert llm.gooseai_api_key.get_secret_value() == "secret-api-key"


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_constructor() -> None:
    llm = GooseAI(gooseai_api_key="secret-api-key")
    assert str(llm.gooseai_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.gooseai_api_key)
    assert "secret-api-key" not in repr(llm)


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_from_env() -> None:
    with MonkeyPatch.context() as mp:
        mp.setenv("GOOSEAI_API_KEY", "secret-api-key")
        llm = GooseAI()
        assert str(llm.gooseai_api_key) == "**********"
        assert "secret-api-key" not in repr(llm.gooseai_api_key)
        assert "secret-api-key" not in repr(llm)
