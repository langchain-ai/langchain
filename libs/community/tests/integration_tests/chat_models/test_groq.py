"""Test ChatGroq chat model."""

from typing import Any

import pytest
from groq import BadRequestError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.groq import ChatGroq
from tests.unit_tests.callbacks.fake_callback_handler import (
    FakeCallbackHandler,
    FakeCallbackHandlerWithChatStart,
)


@pytest.mark.scheduled
def test_chat_groq() -> None:
    """Test Chat wrapper."""
    chat = ChatGroq(
        temperature=0.7,
        base_url=None,
        groq_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_groq_system_message() -> None:
    """Test ChatGroq wrapper with system message."""
    chat = ChatGroq(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_groq_generate() -> None:
    """Test ChatGroq wrapper with generate."""
    n = 1
    chat = ChatGroq(max_tokens=10, n=n)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_groq_multiple_completions() -> None:
    """Test ChatGroq wrapper with multiple completions."""
    chat = ChatGroq(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    # Multiple completions is not currently supported
    with pytest.raises(BadRequestError):
        chat._generate([message])
    #     response = chat._generate([message])
    # assert isinstance(response, ChatResult)
    # assert len(response.generations) == 5
    # for generation in response.generations:
    #     assert isinstance(generation.message, BaseMessage)
    #     assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_groq_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatGroq(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_groq_streaming_generation_info() -> None:
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
    chat = ChatGroq(
        max_tokens=2,
        temperature=0,
        callbacks=[callback],
    )
    list(chat.stream("Respond with the single word Hello"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello"


def test_chat_groq_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatGroq(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_groq_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatGroq(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


@pytest.mark.scheduled
async def test_async_chat_groq() -> None:
    """Test async generation."""
    n = 1
    chat = ChatGroq(max_tokens=10, n=n)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_groq_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandlerWithChatStart()
    chat = ChatGroq(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
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
def test_groq_streaming() -> None:
    """Test streaming tokens from Groq."""
    llm = ChatGroq(max_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_groq_astream() -> None:
    """Test streaming tokens from Groq."""
    llm = ChatGroq(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_groq_abatch() -> None:
    """Test streaming tokens from ChatGroq."""
    llm = ChatGroq(max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_groq_abatch_tags() -> None:
    """Test batch tokens from ChatGroq."""
    llm = ChatGroq(max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_groq_batch() -> None:
    """Test batch tokens from ChatGroq."""
    llm = ChatGroq(max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_groq_ainvoke() -> None:
    """Test invoke tokens from ChatGroq."""
    llm = ChatGroq(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_groq_invoke() -> None:
    """Test invoke tokens from ChatGroq."""
    llm = ChatGroq(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_stream() -> None:
    """Test streaming tokens from Groq."""
    llm = ChatGroq()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from Groq."""
    llm = ChatGroq()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatGroq."""
    llm = ChatGroq()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatGroq."""
    llm = ChatGroq()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatGroq."""
    llm = ChatGroq()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)
