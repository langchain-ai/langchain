"""Test Perplexity API wrapper."""

from typing import Generator

from langchain_core.callbacks import CallbackManager

from langchain_community.llms.perplexity import Perplexity
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_perplexity_model_name_param() -> None:
    llm = PerplexityLLM(model_name="foo")
    assert llm.model == "foo"


def test_perplexity_model_param() -> None:
    llm = PerplexityLLM(model="foo")
    assert llm.model == "foo"


def test_perplexity_call() -> None:
    """Test valid call to Perplexity."""
    llm = PerplexityLLM(model="pplx-70b-online")
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_perplexity_streaming() -> None:
    """Test streaming tokens from Perplexity."""
    llm = PerplexityLLM(model="pplx-70b-online")
    generator = llm.stream("I'm John Doe")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


def test_perplexity_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = PerplexityLLM(
        model="pplx-70b-online",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams > 1
