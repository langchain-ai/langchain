"""Test Cohere API wrapper."""

from pathlib import Path

from langchain.llms.cohere import Cohere
from langchain.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_cohere_call() -> None:
    """Test valid call to cohere."""
    llm = Cohere(max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an Cohere LLM."""
    llm = Cohere(max_tokens=10)
    llm.save(file_path=tmp_path / "cohere.yaml")
    loaded_llm = load_llm(tmp_path / "cohere.yaml")
    assert_llm_equality(llm, loaded_llm)
