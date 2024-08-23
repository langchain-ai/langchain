"""Test Perplexity Chat API wrapper."""

import os

import pytest

from langchain_community.chat_models import ChatPerplexity

os.environ["PPLX_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


@pytest.mark.requires("openai")
def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model="test", model_kwargs={"foo": "bar"})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
def test_perplexity_initialization() -> None:
    """Test perplexity initialization."""
    # Verify that chat perplexity can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatPerplexity(  # type: ignore[call-arg]
            model="test", timeout=1, api_key="test", temperature=0.7, verbose=True
        ),
        ChatPerplexity(  # type: ignore[call-arg]
            model="test",
            request_timeout=1,
            pplx_api_key="test",
            temperature=0.7,
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert model.pplx_api_key == "test"
