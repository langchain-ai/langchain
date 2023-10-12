"""Test AzureChatOpenAI wrapper."""
import os
from typing import Any

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, HumanMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION", "")
os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_API_BASE", "")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "")


@pytest.mark.scheduled
def test_chat_openai() -> None:
    """Test AzureChatOpenAI wrapper."""
    chat = AzureChatOpenAI(max_tokens=10, deployment_name=DEPLOYMENT_NAME)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_openai_generate() -> None:
    """Test AzureChatOpenAI wrapper with generate."""
    chat = AzureChatOpenAI(max_tokens=10, n=2, deployment_name=DEPLOYMENT_NAME)
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
    """Test AzureChatOpenAI wrapper with multiple completions."""
    chat = AzureChatOpenAI(max_tokens=10, n=5, deployment_name=DEPLOYMENT_NAME)
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
    chat = AzureChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
        deployment_name=DEPLOYMENT_NAME,
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
    chat = AzureChatOpenAI(
        max_tokens=2,
        temperature=0,
        callback_manager=callback_manager,
        deployment_name=DEPLOYMENT_NAME,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = AzureChatOpenAI(max_tokens=10, deployment_name=DEPLOYMENT_NAME)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = AzureChatOpenAI(
        max_tokens=10, streaming=True, deployment_name=DEPLOYMENT_NAME
    )
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        AzureChatOpenAI(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
            deployment_name=DEPLOYMENT_NAME,
        )


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = AzureChatOpenAI(max_tokens=10, n=2, deployment_name=DEPLOYMENT_NAME)
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
    chat = AzureChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
        deployment_name=DEPLOYMENT_NAME,
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
    llm = AzureChatOpenAI(foo=3, max_tokens=10, deployment_name=DEPLOYMENT_NAME)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = AzureChatOpenAI(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        AzureChatOpenAI(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        AzureChatOpenAI(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        AzureChatOpenAI(model_kwargs={"model": "text-davinci-003"})


@pytest.fixture
def llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(max_tokens=10, deployment_name=DEPLOYMENT_NAME)


@pytest.mark.scheduled
def test_openai_streaming(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_astream(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_abatch(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from AzureChatOpenAI."""

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_abatch_tags(llm: AzureChatOpenAI) -> None:
    """Test batch tokens from AzureChatOpenAI."""

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch(llm: AzureChatOpenAI) -> None:
    """Test batch tokens from AzureChatOpenAI."""

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_openai_ainvoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_openai_invoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
