"""Test Tongyi API wrapper."""
from langchain.llms.tongyi import Tongyi
from langchain.schema import LLMResult


def test_tongyi_call() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm("who are you")
    assert isinstance(output, str)


def test_tongyi_generate() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_tongyi_generate_stream() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi(streaming=True)
    output = llm.generate(["who are you"])
    print(output)
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
