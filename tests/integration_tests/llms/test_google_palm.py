"""Test Google PaLM Text API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

from pathlib import Path

from langchain.llms.google_palm import GooglePalm
from langchain.llms.loading import load_llm


def test_google_palm_call() -> None:
    """Test valid call to Google PaLM text API."""
    llm = GooglePalm(max_output_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading a Google PaLM LLM."""
    llm = GooglePalm(max_output_tokens=10)
    llm.save(file_path=tmp_path / "google_palm.yaml")
    loaded_llm = load_llm(tmp_path / "google_palm.yaml")
    assert loaded_llm == llm
