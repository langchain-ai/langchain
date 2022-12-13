"""Test AI21 API wrapper."""

from pathlib import Path

from langchain.llms.ai21 import AI21
from langchain.llms.loading import load_llm


def test_ai21_call() -> None:
    """Test valid call to ai21."""
    llm = AI21(maxTokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an NLPCloud LLM."""
    llm = AI21(maxTokens=10)
    llm.save(file_path=tmp_path / "nlpcloud.yaml")
    loaded_llm = load_llm(tmp_path / "nlpcloud.yaml")
    assert llm == loaded_llm
