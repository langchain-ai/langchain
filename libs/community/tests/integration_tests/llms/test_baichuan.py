"""Test Baichuan LLM Endpoint."""
from langchain_core.outputs import LLMResult

from langchain_community.llms.baichuan import BaichuanLLM


def test_call() -> None:
    """Test valid call to baichuan."""
    llm = BaichuanLLM()
    output = llm("Who won the second world war?")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to baichuan."""
    llm = BaichuanLLM()
    output = llm.generate(["Who won the second world war?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
