"""Test Cohere API wrapper."""

from pathlib import Path

from pydantic import SecretStr
from pytest import MonkeyPatch

from langchain_community.llms.cohere import Cohere
from langchain_community.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_cohere_call() -> None:
    """Test valid call to cohere."""
    llm = Cohere(max_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_cohere_api_key(monkeypatch: MonkeyPatch) -> None:
    """Test that cohere api key is a secret key."""
    # test initialization from init
    assert isinstance(Cohere(cohere_api_key="1").cohere_api_key, SecretStr)  # type: ignore[arg-type]

    # test initialization from env variable
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    assert isinstance(Cohere().cohere_api_key, SecretStr)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an Cohere LLM."""
    llm = Cohere(max_tokens=10)
    llm.save(file_path=tmp_path / "cohere.yaml")
    loaded_llm = load_llm(tmp_path / "cohere.yaml")
    assert_llm_equality(llm, loaded_llm)
