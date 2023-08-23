"""Test ChatOpenAI wrapper."""


from typing import Any

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_openai() -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_openai_model() -> None:
    """Test ChatOpenAI wrapper handles model_name."""
    chat = ChatOpenAI(model="foo")
    assert chat.model_name == "foo"
    chat = ChatOpenAI(model_name="bar")
    assert chat.model_name == "bar"


def test_chat_openai_system_message() -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_openai_generate() -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatOpenAI(max_tokens=10, n=2)
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
def test_chat_openai_multiple_completions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_openai_streaming_generation_info() -> None:
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
    chat = ChatOpenAI(
        max_tokens=2,
        temperature=0,
        callback_manager=callback_manager,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatOpenAI(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = ChatOpenAI(max_tokens=10, n=2)
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
async def test_async_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
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


def test_chat_openai_extra_kwargs() -> None:
    """Test extra kwargs to chat openai."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatOpenAI(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatOpenAI(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatOpenAI(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatOpenAI(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatOpenAI(model_kwargs={"model": "text-davinci-003"})


@pytest.mark.scheduled
def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_abatch() -> None:
    """Test streaming tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_ainvoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_openai_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
