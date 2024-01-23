"""Test ChatDeepInfra wrapper."""
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.deepinfra import ChatDeepInfra
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_chat_deepinfra() -> None:
    """Test valid call to DeepInfra."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_deepinfra_streaming() -> None:
    callback_handler = FakeCallbackHandler()
    chat = ChatDeepInfra(
        callbacks=[callback_handler],
        streaming=True,
        max_tokens=10,
    )
    response = chat.invoke([HumanMessage(content="Hello")])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


async def test_async_chat_deepinfra() -> None:
    """Test async generation."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 1
    generation = response.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.text, str)
    assert generation.text == generation.message.content


async def test_async_chat_deepinfra_streaming() -> None:
    callback_handler = FakeCallbackHandler()
    chat = ChatDeepInfra(
        # model="meta-llama/Llama-2-7b-chat-hf",
        callbacks=[callback_handler],
        max_tokens=10,
        streaming=True,
        timeout=5,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 1
    generation = response.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.text, str)
    assert generation.text == generation.message.content
