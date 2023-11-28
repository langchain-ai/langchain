"""Test LlamaChat wrapper."""

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.nv_aiplay import LlamaChat, NVAIPlayChat
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_aiplay() -> None:
    """Test NVAIPlayChat wrapper."""
    chat = LlamaChat(
        temperature=0.7,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_aiplay_model() -> None:
    """Test LlamaChat wrapper handles model_name."""
    chat = NVAIPlayChat(model_name="mistral")
    assert chat.model_name == "mistral"


def test_chat_aiplay_system_message() -> None:
    """Test LlamaChat wrapper with system message."""
    chat = LlamaChat(max_tokens=36)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


## TODO: Not sure if we want to support the n syntax. Trash or keep test


@pytest.mark.scheduled
def test_chat_aiplay_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = NVAIPlayChat(
        max_tokens=36,
        streaming=True,
        temperature=0.1,
        callbacks=callback_manager,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_aiplay_streaming() -> None:
    """Test streaming tokens from aiplay."""
    llm = LlamaChat(max_tokens=36)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_astream() -> None:
    """Test streaming tokens from aiplay."""
    llm = LlamaChat(max_tokens=35)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch() -> None:
    """Test streaming tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=36)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch_tags() -> None:
    """Test batch tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=55)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_aiplay_batch() -> None:
    """Test batch tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_ainvoke() -> None:
    """Test invoke tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_aiplay_invoke() -> None:
    """Test invoke tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
