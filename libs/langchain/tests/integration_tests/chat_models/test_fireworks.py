"""Test ChatFireworks wrapper."""
import sys

import pytest

from langchain.chat_models.fireworks import ChatFireworks
from langchain.schema import ChatGeneration, ChatResult, LLMResult
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage

if sys.version_info < (3, 9):
    pytest.skip("fireworks-ai requires Python > 3.8", allow_module_level=True)


@pytest.fixture
def chat() -> ChatFireworks:
    return ChatFireworks(model_kwargs={"temperature": 0, "max_tokens": 512})


@pytest.mark.scheduled
def test_chat_fireworks(chat: ChatFireworks) -> None:
    """Test ChatFireworks wrapper."""
    message = HumanMessage(content="What is the weather in Redwood City, CA today")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_fireworks_model() -> None:
    """Test ChatFireworks wrapper handles model_name."""
    chat = ChatFireworks(model="foo")
    assert chat.model == "foo"


@pytest.mark.scheduled
def test_chat_fireworks_system_message(chat: ChatFireworks) -> None:
    """Test ChatFireworks wrapper with system message."""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_fireworks_generate() -> None:
    """Test ChatFireworks wrapper with generate."""
    chat = ChatFireworks(model_kwargs={"n": 2})
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_fireworks_multiple_completions() -> None:
    """Test ChatFireworks wrapper with multiple completions."""
    chat = ChatFireworks(model_kwargs={"n": 5})
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_fireworks_llm_output_contains_model_id(chat: ChatFireworks) -> None:
    """Test llm_output contains model_id."""
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model"] == chat.model


@pytest.mark.scheduled
def test_fireworks_invoke(chat: ChatFireworks) -> None:
    """Tests chat completion with invoke"""
    result = chat.invoke("How is the weather in New York today?", stop=[","])
    assert isinstance(result.content, str)
    assert result.content[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_ainvoke(chat: ChatFireworks) -> None:
    """Tests chat completion with invoke"""
    result = await chat.ainvoke("How is the weather in New York today?", stop=[","])
    assert isinstance(result.content, str)
    assert result.content[-1] == ","


@pytest.mark.scheduled
def test_fireworks_batch(chat: ChatFireworks) -> None:
    """Test batch tokens from ChatFireworks."""
    result = chat.batch(
        [
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
        ],
        config={"max_concurrency": 5},
        stop=[","],
    )
    for token in result:
        assert isinstance(token.content, str)
        assert token.content[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_abatch(chat: ChatFireworks) -> None:
    """Test batch tokens from ChatFireworks."""
    result = await chat.abatch(
        [
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
            "What is the weather in Redwood City, CA today",
        ],
        config={"max_concurrency": 5},
        stop=[","],
    )
    for token in result:
        assert isinstance(token.content, str)
        assert token.content[-1] == ","


@pytest.mark.scheduled
def test_fireworks_streaming(chat: ChatFireworks) -> None:
    """Test streaming tokens from Fireworks."""

    for token in chat.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_fireworks_streaming_stop_words(chat: ChatFireworks) -> None:
    """Test streaming tokens with stop words."""

    last_token = ""
    for token in chat.stream("I'm Pickle Rick", stop=[","]):
        last_token = token.content
        assert isinstance(token.content, str)
    assert last_token[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_chat_fireworks_agenerate() -> None:
    """Test ChatFireworks wrapper with generate."""
    chat = ChatFireworks(model_kwargs={"n": 2})
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_astream(chat: ChatFireworks) -> None:
    """Test streaming tokens from Fireworks."""

    last_token = ""
    async for token in chat.astream(
        "Who's the best quarterback in the NFL?", stop=[","]
    ):
        last_token = token.content
        assert isinstance(token.content, str)
    assert last_token[-1] == ","
