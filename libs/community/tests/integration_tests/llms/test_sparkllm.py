"""Test SparkLLM."""

from langchain_core.outputs import LLMResult

from langchain_community.llms.sparkllm import SparkLLM


def test_call() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
