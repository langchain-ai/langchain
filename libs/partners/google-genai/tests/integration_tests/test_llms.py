"""Test Google GenerativeAI API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

import pytest
from langchain_core.outputs import LLMResult

from langchain_google_genai.llms import GoogleGenerativeAI

model_names = ["models/text-bison-001", "gemini-pro"]


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GoogleGenerativeAI(max_output_tokens=10, model=model_name)
    else:
        llm = GoogleGenerativeAI(max_output_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_palm"
    if model_name and "gemini" in model_name:
        assert llm.client.model_name == "models/gemini-pro"
    else:
        assert llm.model == "models/text-bison-001"


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_generate(model_name: str) -> None:
    n = 1 if model_name == "gemini-pro" else 2
    llm = GoogleGenerativeAI(temperature=0.3, n=n, model=model_name)
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == n


def test_google_generativeai_get_num_tokens() -> None:
    llm = GoogleGenerativeAI(model="models/text-bison-001")
    output = llm.get_num_tokens("How are you?")
    assert output == 4


async def test_google_generativeai_agenerate() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="gemini-pro")
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


def test_generativeai_stream() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="gemini-pro")
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


def test_generativeai_get_num_tokens_gemini() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="gemini-pro")
    output = llm.get_num_tokens("How are you?")
    assert output == 4
