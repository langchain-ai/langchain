"""Test Google GenerativeAI API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

from langchain_community.llms.google_palm import GoogleGenerativeAI
from langchain_community.llms.loading import load_llm

model_names = [None, "models/text-bison-001", "gemini-pro"]


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GoogleGenerativeAI(max_output_tokens=10, model_name=model_name)
    else:
        llm = GoogleGenerativeAI(max_output_tokens=10)
    output = llm("Say foo:")
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
        llm = GoogleGenerativeAI(temperature=0.3, n=n, model_name=model_name)
    else:
        llm = GoogleGenerativeAI(temperature=0.3, n=n)
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == n


def test_google_generativeai_get_num_tokens() -> None:
    llm = GoogleGenerativeAI()
    output = llm.get_num_tokens("How are you?")
    assert output == 4


async def test_google_generativeai_agenerate() -> None:
    llm = GoogleGenerativeAI(temperature=0, model_name="gemini-pro")
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


def test_generativeai_stream() -> None:
    llm = GoogleGenerativeAI(temperature=0, model_name="gemini-pro")
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading a Google PaLM LLM."""
    llm = GoogleGenerativeAI(max_output_tokens=10)
    llm.save(file_path=tmp_path / "google_palm.yaml")
    loaded_llm = load_llm(tmp_path / "google_palm.yaml")
    assert loaded_llm == llm


def test_invoke_multimodal() -> None:
    llm = GoogleGenerativeAI(model_name="gemini-pro-vision")
    gcs_url = (
        "gs://cloud-samples-data/generative-ai/image/"
        "320px-Felis_catus-cat_on_snow.jpg"
    )
    image_message = {
        "type": "image_url",
        "image_url": {"url": gcs_url, "mime_type": "image/jpeg"},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    message = HumanMessage(content=[text_message, image_message])
    output = llm.invoke([message])
    assert isinstance(output, str)
