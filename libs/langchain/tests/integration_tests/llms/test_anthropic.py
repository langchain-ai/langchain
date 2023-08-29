"""Test Anthropic API wrapper."""
from typing import Generator

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.llms.anthropic import Anthropic
from langchain.schema import LLMResult
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = Anthropic(model_name="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = Anthropic(model="foo")
    assert llm.model == "foo"


def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    llm = Anthropic(model="claude-instant-1")
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = Anthropic(model="claude-instant-1")
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


def test_anthropic_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = Anthropic(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams > 1


@pytest.mark.asyncio
async def test_anthropic_async_generate() -> None:
    """Test async generate."""
    llm = Anthropic()
    output = await llm.agenerate(["How many toes do dogs have?"])
    assert isinstance(output, LLMResult)


@pytest.mark.asyncio
async def test_anthropic_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = Anthropic(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["How many toes do dogs have?"])
    assert callback_handler.llm_streams > 1
    assert isinstance(result, LLMResult)
