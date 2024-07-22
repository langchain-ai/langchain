"""Test ChatAI21 chat model."""

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
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
    llm = ChatAI21(model=model)  # type: ignore[call-arg]

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


@pytest.mark.parametrize(
    ids=[
        "when_j2_model_num_results_is_1",
        "when_j2_model_num_results_is_3",
        "when_jamba_model_n_is_1",
        "when_jamba_model_n_is_3",
    ],
    argnames=["model", "num_results"],
    argvalues=[
        (J2_CHAT_MODEL_NAME, 1),
        (J2_CHAT_MODEL_NAME, 3),
        (JAMBA_CHAT_MODEL_NAME, 1),
        (JAMBA_CHAT_MODEL_NAME, 3),
    ],
)
def test_generation(model: str, num_results: int) -> None:
    """Test generation with multiple models and different result counts."""
    # Determine the configuration key based on the model type
    config_key = "n" if model == JAMBA_CHAT_MODEL_NAME else "num_results"

    # Create the model instance using the appropriate key for the result count
    llm = ChatAI21(model=model, **{config_key: num_results})  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]

    message = HumanMessage(content="Hello, this is a test. Can you help me please?")

    result = llm.generate([[message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == num_results
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
    llm = ChatAI21(model=model)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")

    result = await llm.agenerate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test__chat_stream() -> None:
    llm = ChatAI21(model="jamba-instruct")  # type: ignore[call-arg]
    message = HumanMessage(content="What is the meaning of life?")

    for chunk in llm.stream([message]):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)


def test__j2_chat_stream__should_raise_error() -> None:
    llm = ChatAI21(model="j2-ultra")  # type: ignore[call-arg]
    message = HumanMessage(content="What is the meaning of life?")

    with pytest.raises(NotImplementedError):
        for _ in llm.stream([message]):
            pass
