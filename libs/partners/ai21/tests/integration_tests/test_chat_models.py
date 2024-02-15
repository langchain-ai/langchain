"""Test ChatAI21 chat model."""
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration

from langchain_ai21.chat_models import ChatAI21


def test_invoke() -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model="j2-ultra")

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_generation() -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model="j2-ultra")
    message = HumanMessage(content="Hello")

    result = llm.generate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


async def test_ageneration() -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model="j2-ultra")
    message = HumanMessage(content="Hello")

    result = await llm.agenerate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
