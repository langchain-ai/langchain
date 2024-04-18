"""Test ChatAI21 chat model."""
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration

from langchain_ai21.chat_models import ChatAI21
from tests.unit_tests.conftest import J2_CHAT_MODEL_NAME, JAMBA_CHAT_MODEL_NAME


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model"],
    argvalues=[
        (J2_CHAT_MODEL_NAME,),
        (JAMBA_CHAT_MODEL_NAME,),
    ],
)
def test_invoke(model: str) -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model=model)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model"],
    argvalues=[
        (J2_CHAT_MODEL_NAME,),
        (JAMBA_CHAT_MODEL_NAME,),
    ],
)
def test_generation(model: str) -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model=model)
    message = HumanMessage(content="Hello")

    result = llm.generate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model"],
    argvalues=[
        (J2_CHAT_MODEL_NAME,),
        (JAMBA_CHAT_MODEL_NAME,),
    ],
)
async def test_ageneration(model: str) -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model=model)
    message = HumanMessage(content="Hello")

    result = await llm.agenerate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
