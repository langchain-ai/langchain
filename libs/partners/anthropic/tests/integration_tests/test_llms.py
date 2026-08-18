"""Test Anthropic API wrapper."""

import os
from collections.abc import Generator

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult

from langchain_anthropic import AnthropicLLM
from tests.unit_tests._utils import FakeCallbackHandler

MODEL = "claude-sonnet-4-5-20250929"

# The deprecated `AnthropicLLM` class has no LangSmith gateway support, so it
# cannot authenticate when requests are routed through the gateway. Skip the
# network-calling tests in that case; they still run against a direct
# `ANTHROPIC_API_KEY`. Mirrors the gateway truthiness in `langchain_core`.
_GATEWAY_ENABLED = (os.getenv("LANGSMITH_GATEWAY") or "").lower() not in (
    "",
    "false",
    "0",
    "no",
)
_skip_under_gateway = pytest.mark.skipif(
    _GATEWAY_ENABLED,
    reason="AnthropicLLM is deprecated and not compatible with the LangSmith gateway",
)


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = AnthropicLLM(model_name="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = AnthropicLLM(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


@_skip_under_gateway
def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@_skip_under_gateway
def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@_skip_under_gateway
def test_anthropic_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = AnthropicLLM(
        model=MODEL,  # type: ignore[call-arg]
        streaming=True,
        callbacks=callback_manager,
        verbose=True,
    )
    llm.invoke("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams > 1


@_skip_under_gateway
async def test_anthropic_async_generate() -> None:
    """Test async generate."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    output = await llm.agenerate(["How many toes do dogs have?"])
    assert isinstance(output, LLMResult)


@_skip_under_gateway
async def test_anthropic_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = AnthropicLLM(
        model=MODEL,  # type: ignore[call-arg]
        streaming=True,
        callbacks=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["How many toes do dogs have?"])
    assert callback_handler.llm_streams > 1
    assert isinstance(result, LLMResult)
