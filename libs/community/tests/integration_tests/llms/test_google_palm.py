"""Test Google GenerativeAI API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

from pathlib import Path

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.llms.google_palm import GooglePalm
from langchain_community.llms.loading import load_llm

model_names = [None, "models/text-bison-001", "gemini-pro"]


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GooglePalm(max_output_tokens=10, model_name=model_name)  # type: ignore[call-arg]
    else:
        llm = GooglePalm(max_output_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_palm"
    if model_name and "gemini" in model_name:
        assert llm.client.model_name == "models/gemini-pro"
    else:
        assert llm.model_name == "models/text-bison-001"


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_generate(model_name: str) -> None:
    n = 1 if model_name == "gemini-pro" else 2
    if model_name:
        llm = GooglePalm(temperature=0.3, n=n, model_name=model_name)  # type: ignore[call-arg]
    else:
        llm = GooglePalm(temperature=0.3, n=n)  # type: ignore[call-arg]
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == n


def test_google_generativeai_get_num_tokens() -> None:
    llm = GooglePalm()  # type: ignore[call-arg]
    output = llm.get_num_tokens("How are you?")
    assert output == 4


async def test_google_generativeai_agenerate() -> None:
    llm = GooglePalm(temperature=0, model_name="gemini-pro")  # type: ignore[call-arg]
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


def test_generativeai_stream() -> None:
    llm = GooglePalm(temperature=0, model_name="gemini-pro")  # type: ignore[call-arg]
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading a Google PaLM LLM."""
    llm = GooglePalm(max_output_tokens=10)  # type: ignore[call-arg]
    llm.save(file_path=tmp_path / "google_palm.yaml")
    loaded_llm = load_llm(tmp_path / "google_palm.yaml")
    assert loaded_llm == llm
