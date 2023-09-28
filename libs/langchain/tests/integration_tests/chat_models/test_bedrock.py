"""Test Bedrock chat model."""
from typing import List

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import BedrockChat
from langchain.schema import ChatGeneration, LLMResult
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_bedrock() -> None:
    """Test BedrockChat wrapper."""
    chat = BedrockChat()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_bedrock_generate() -> None:
    """Test BedrockChat wrapper with generate."""
    chat = BedrockChat()
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
def test_chat_bedrock_multiple_completions() -> None:
    """Test BedrockChat wrapper with multiple completions."""
    chat = BedrockChat()
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_bedrock_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = BedrockChat(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_bedrock_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    callback_manager = CallbackManager([callback])
    chat = BedrockChat(
        callback_manager=callback_manager,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_async_chat_bedrock() -> None:
    """Test async generation."""
    chat = BedrockChat()
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_async_chat_bedrock_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = BedrockChat(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_bedrock_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = BedrockChat()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = BedrockChat()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_abatch() -> None:
    """Test streaming tokens from BedrockChat."""
    llm = BedrockChat()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_abatch_tags() -> None:
    """Test batch tokens from BedrockChat."""
    llm = BedrockChat()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_bedrock_batch() -> None:
    """Test batch tokens from BedrockChat."""
    llm = BedrockChat()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_ainvoke() -> None:
    """Test invoke tokens from BedrockChat."""
    llm = BedrockChat()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_bedrock_invoke() -> None:
    """Test invoke tokens from BedrockChat."""
    llm = BedrockChat()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
