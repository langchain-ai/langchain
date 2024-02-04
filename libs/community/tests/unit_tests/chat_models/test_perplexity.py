"""Test Perplexity Chat API wrapper."""
import os

import pytest

from langchain_community.chat_models import ChatPerplexity

os.environ["PPLX_API_KEY"] = "foo"


def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model_name="foo")
    assert llm.model == "foo"


def test_perplexity_model_param() -> None:
    llm = ChatPerplexity(model="foo")
    assert llm.model == "foo"


def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_perplexity_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ChatPerplexity(model_kwargs={"max_tokens_to_sample": 5})


def test_perplexity_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatPerplexity(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


def test_perplexity_initialization() -> None:
    """Test perplexity initialization."""
    # Verify that chat perplexity can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatPerplexity(model="test", perplexity_api_key="test")
