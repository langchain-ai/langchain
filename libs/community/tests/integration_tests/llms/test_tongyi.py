"""Test Tongyi API wrapper."""

from langchain_core.outputs import LLMResult

from langchain_community.llms.tongyi import Tongyi


def test_tongyi_call() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm.invoke("who are you")
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
    print(output)  # noqa: T201
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_tongyi_with_param_alias() -> None:
    """Test tongyi parameters alias"""
    llm = Tongyi(model="qwen-max", api_key="your-api_key")
    assert llm.model_name == "qwen-max"
    assert llm.dashscope_api_key == "your-api_key"
