"""Test ChatGLM API wrapper."""

from langchain_core.outputs import LLMResult

from langchain_community.llms.chatglm import ChatGLM


def test_chatglm_call() -> None:
    """Test valid call to chatglm."""
    llm = ChatGLM()
    output = llm.invoke("北京和上海这两座城市有什么不同？")
    assert isinstance(output, str)


def test_chatglm_generate() -> None:
    """Test valid call to chatglm."""
    llm = ChatGLM()
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
