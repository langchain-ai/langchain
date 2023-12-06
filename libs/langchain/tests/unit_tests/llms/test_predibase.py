"""Test Predibase"""
import pytest
from langchain_core.pydantic_v1 import SecretStr

from langchain.llms.predibase import Predibase


@pytest.mark.requires("predibase")
def test_api_key_is_secret_string() -> None:
    llm = Predibase(model="test", predibase_api_key="secret-api-key")
    assert isinstance(llm.predibase_api_key, SecretStr)


@pytest.mark.requires("predibase")
def test_api_key_masked_when_passed_via_constructor() -> None:
    llm = Predibase(model="test", predibase_api_key="secret-api-key")
    assert str(llm.predibase_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.predibase_api_key)
    assert "secret-api-key" not in repr(llm)
