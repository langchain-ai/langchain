"""Test volc engine maas LLM model."""

from typing import Generator

from langchain.llms.volcengine_maas import VolcEngineMaasLLM
from langchain.schema import LLMResult


def test_default_call() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()
    output = llm("tell me a joke")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()
    output = llm.generate(["tell me a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_stream() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM(streaming=True)
    output = llm.stream("tell me a joke")
    assert isinstance(output, Generator)
