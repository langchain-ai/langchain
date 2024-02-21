"""Test BigDL LLM"""
from langchain_core.outputs import LLMResult

from langchain_community.llms.bigdl import BigdlLLM

def test_call() -> None:
    """Test valid call to baichuan."""
    llm = BigdlLLM()
    output = llm("Who won the second world war?")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to baichuan."""
    llm = BigdlLLM()
    output = llm.generate(["Who won the second world war?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
