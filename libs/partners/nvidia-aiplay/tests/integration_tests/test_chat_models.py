"""Test ChatNVAIPlay chat model."""
from nvidia_aiplay.chat_models import GeneralChat, NVAIPlayChat

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_aiplay() -> None:
    """Test NVAIPlayChat wrapper."""
    chat = GeneralChat(
        temperature=0.7,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_aiplay_model() -> None:
    """Test GeneralChat wrapper handles model."""
    chat = NVAIPlayChat(model="mistral")
    assert chat.model == "mistral"


def test_chat_aiplay_system_message() -> None:
    """Test GeneralChat wrapper with system message."""
    chat = GeneralChat(max_tokens=36)
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
    llm = GeneralChat(max_tokens=36)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_astream() -> None:
    """Test streaming tokens from aiplay."""
    llm = GeneralChat(max_tokens=35)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch() -> None:
    """Test streaming tokens from GeneralChat."""
    llm = GeneralChat(max_tokens=36)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch_tags() -> None:
    """Test batch tokens from GeneralChat."""
    llm = GeneralChat(max_tokens=55)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_aiplay_batch() -> None:
    """Test batch tokens from GeneralChat."""
    llm = GeneralChat(max_tokens=60)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_ainvoke() -> None:
    """Test invoke tokens from GeneralChat."""
    llm = GeneralChat(max_tokens=60)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_aiplay_invoke() -> None:
    """Test invoke tokens from GeneralChat."""
    llm = GeneralChat(max_tokens=60)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


## Default methods
# def test_integration_stream() -> None:
#     """Test streaming tokens from OpenAI."""
#     llm = ChatNVAIPlay()

#     for token in llm.stream("I'm Pickle Rick"):
#         assert isinstance(token.content, str)


# async def test_integration_astream() -> None:
#     """Test streaming tokens from OpenAI."""
#     llm = ChatNVAIPlay()

#     async for token in llm.astream("I'm Pickle Rick"):
#         assert isinstance(token.content, str)


# async def test_integration_abatch() -> None:
#     """Test streaming tokens from ChatNVAIPlay."""
#     llm = ChatNVAIPlay()

#     result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
#     for token in result:
#         assert isinstance(token.content, str)


# async def test_integration_abatch_tags() -> None:
#     """Test batch tokens from ChatNVAIPlay."""
#     llm = ChatNVAIPlay()

#     result = await llm.abatch(
#         ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
#     )
#     for token in result:
#         assert isinstance(token.content, str)


# def test_integration_batch() -> None:
#     """Test batch tokens from ChatNVAIPlay."""
#     llm = ChatNVAIPlay()

#     result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
#     for token in result:
#         assert isinstance(token.content, str)


# async def test_integration_ainvoke() -> None:
#     """Test invoke tokens from ChatNVAIPlay."""
#     llm = ChatNVAIPlay()

#     result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
#     assert isinstance(result.content, str)


# def test_integration_invoke() -> None:
#     """Test invoke tokens from ChatNVAIPlay."""
#     llm = ChatNVAIPlay()

#     result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
#     assert isinstance(result.content, str)
