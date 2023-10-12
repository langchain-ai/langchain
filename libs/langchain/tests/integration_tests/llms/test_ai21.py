"""Test AI21 API wrapper."""

from pathlib import Path

from langchain.llms.ai21 import AI21
from langchain.llms.loading import load_llm


def test_ai21_call() -> None:
    """Test valid call to ai21."""
    llm = AI21(maxTokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_ai21_call_experimental() -> None:
    """Test valid call to ai21 with an experimental model."""
    llm = AI21(maxTokens=10, model="j1-grande-instruct")
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an AI21 LLM."""
    llm = AI21(maxTokens=10)
    llm.save(file_path=tmp_path / "ai21.yaml")
    loaded_llm = load_llm(tmp_path / "ai21.yaml")
    assert llm == loaded_llm
