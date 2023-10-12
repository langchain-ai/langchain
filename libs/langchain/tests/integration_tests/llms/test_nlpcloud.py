"""Test NLPCloud API wrapper."""

from pathlib import Path

from langchain.llms.loading import load_llm
from langchain.llms.nlpcloud import NLPCloud
from tests.integration_tests.llms.utils import assert_llm_equality


def test_nlpcloud_call() -> None:
    """Test valid call to nlpcloud."""
    llm = NLPCloud(max_length=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an NLPCloud LLM."""
    llm = NLPCloud(max_length=10)
    llm.save(file_path=tmp_path / "nlpcloud.yaml")
    loaded_llm = load_llm(tmp_path / "nlpcloud.yaml")
    assert_llm_equality(llm, loaded_llm)
