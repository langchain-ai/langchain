"""Test Bedrock chat model."""
from typing import Any

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import BedrockChat
from langchain.schema import ChatGeneration, LLMResult
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.fixture
def chat() -> BedrockChat:
    return BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0})


@pytest.mark.scheduled
def test_chat_bedrock(chat: BedrockChat) -> None:
    """Test BedrockChat wrapper."""
    system = SystemMessage(content="You are a helpful assistant.")
    human = HumanMessage(content="Hello")
    response = chat([system, human])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_bedrock_generate(chat: BedrockChat) -> None:
    """Test BedrockChat wrapper with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_bedrock_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = BedrockChat(
        model_id="anthropic.claude-v2",
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
        model_id="anthropic.claude-v2",
        callback_manager=callback_manager,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == " Hello!"


@pytest.mark.scheduled
def test_bedrock_streaming(chat: BedrockChat) -> None:
    """Test streaming tokens from OpenAI."""

    for token in chat.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_astream(chat: BedrockChat) -> None:
    """Test streaming tokens from OpenAI."""

    async for token in chat.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_abatch(chat: BedrockChat) -> None:
    """Test streaming tokens from BedrockChat."""
    result = await chat.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_abatch_tags(chat: BedrockChat) -> None:
    """Test batch tokens from BedrockChat."""
    result = await chat.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_bedrock_batch(chat: BedrockChat) -> None:
    """Test batch tokens from BedrockChat."""
    result = chat.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_bedrock_ainvoke(chat: BedrockChat) -> None:
    """Test invoke tokens from BedrockChat."""
    result = await chat.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_bedrock_invoke(chat: BedrockChat) -> None:
    """Test invoke tokens from BedrockChat."""
    result = chat.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
