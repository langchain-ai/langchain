"""Test ChatAI21 chat model."""

import pytest
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_ai21.chat_models import ChatAI21
from tests.unit_tests.conftest import (
    J2_CHAT_MODEL_NAME,
    JAMBA_1_5_LARGE_CHAT_MODEL_NAME,
    JAMBA_1_5_MINI_CHAT_MODEL_NAME,
    JAMBA_CHAT_MODEL_NAME,
    JAMBA_FAMILY_MODEL_NAMES,
)

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
        "when_jamba1.5-mini_model",
        "when_jamba1.5-large_model",
    ],
    argnames=["model"],
    argvalues=[
        (J2_CHAT_MODEL_NAME,),
        (JAMBA_CHAT_MODEL_NAME,),
        (JAMBA_1_5_MINI_CHAT_MODEL_NAME,),
        (JAMBA_1_5_LARGE_CHAT_MODEL_NAME,),
    ],
)
def test_invoke(model: str) -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model=model, rate_limiter=rate_limiter)  # type: ignore[call-arg]

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


@pytest.mark.parametrize(
    ids=[
        "when_j2_model_num_results_is_1",
        "when_j2_model_num_results_is_3",
        "when_jamba_model_n_is_1",
        "when_jamba_model_n_is_3",
        "when_jamba1.5_mini_model_n_is_1",
        "when_jamba1.5_mini_model_n_is_3",
        "when_jamba1.5_large_model_n_is_1",
        "when_jamba1.5_large_model_n_is_3",
    ],
    argnames=["model", "num_results"],
    argvalues=[
        (J2_CHAT_MODEL_NAME, 1),
        (J2_CHAT_MODEL_NAME, 3),
        (JAMBA_CHAT_MODEL_NAME, 1),
        (JAMBA_CHAT_MODEL_NAME, 3),
        (JAMBA_1_5_MINI_CHAT_MODEL_NAME, 1),
        (JAMBA_1_5_MINI_CHAT_MODEL_NAME, 3),
        (JAMBA_1_5_LARGE_CHAT_MODEL_NAME, 1),
        (JAMBA_1_5_LARGE_CHAT_MODEL_NAME, 3),
    ],
)
def test_generation(model: str, num_results: int) -> None:
    """Test generation with multiple models and different result counts."""
    # Determine the configuration key based on the model type
    config_key = "n" if model in JAMBA_FAMILY_MODEL_NAMES else "num_results"

    # Create the model instance using the appropriate key for the result count
    llm = ChatAI21(model=model, rate_limiter=rate_limiter, **{config_key: num_results})  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]

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
        "when_jamba1.5_mini_model",
        "when_jamba1.5_large_model",
    ],
    argnames=["model"],
    argvalues=[
        (J2_CHAT_MODEL_NAME,),
        (JAMBA_CHAT_MODEL_NAME,),
        (JAMBA_1_5_MINI_CHAT_MODEL_NAME,),
        (JAMBA_1_5_LARGE_CHAT_MODEL_NAME,),
    ],
)
async def test_ageneration(model: str) -> None:
    """Test invoke tokens from AI21."""
    llm = ChatAI21(model=model, rate_limiter=rate_limiter)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")

    result = await llm.agenerate([[message], [message]], config=dict(tags=["foo"]))

    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test__chat_stream() -> None:
    llm = ChatAI21(model="jamba-1.5-mini")  # type: ignore[call-arg]
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


@pytest.mark.parametrize(
    ids=[
        "when_jamba1.5_mini_model",
        "when_jamba1.5_large_model",
    ],
    argnames=["model"],
    argvalues=[
        (JAMBA_1_5_MINI_CHAT_MODEL_NAME,),
        (JAMBA_1_5_LARGE_CHAT_MODEL_NAME,),
    ],
)
def test_tool_calls(model: str) -> None:
    @tool
    def get_weather(location: str, date: str) -> str:
        """“Provide the weather for the specified location on the given date.”"""
        if location == "New York" and date == "2024-12-05":
            return "25 celsius"
        return "32 celsius"

    llm = ChatAI21(model=model, temperature=0)  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])

    chat_messages = [
        SystemMessage(
            content="You are a helpful assistant. "
            "You can use the provided tools "
            "to assist with various tasks and provide "
            "accurate information"
        ),
        HumanMessage(
            content="What is the forecast for the weather "
            "in New York on December 5, 2024?"
        ),
    ]

    response = llm_with_tools.invoke(chat_messages)
    chat_messages.append(response)
    assert response.tool_calls is not None  # type: ignore[attr-defined]
    tool_call = response.tool_calls[0]  # type: ignore[attr-defined]
    assert tool_call["name"] == "get_weather"

    weather = get_weather.invoke(  # type: ignore[attr-defined]
        {"location": tool_call["args"]["location"], "date": tool_call["args"]["date"]}
    )
    chat_messages.append(ToolMessage(content=weather, tool_call_id=tool_call["id"]))
    llm_answer = llm_with_tools.invoke(chat_messages)
    content = llm_answer.content.lower()  # type: ignore[union-attr]
    assert "new york" in content and "25" in content and "celsius" in content
