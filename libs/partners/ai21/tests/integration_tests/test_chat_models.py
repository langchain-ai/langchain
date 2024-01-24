"""Test ChatAI21 chat model."""
import pytest

from langchain_ai21.chat_models import ChatAI21
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration


@pytest.mark.requires("ai21")
def test_invoke() -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


@pytest.mark.requires("ai21")
def test_generation() -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21()
    message = HumanMessage(content="Hello")

    result = llm.generate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
