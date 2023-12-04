"""Test Anthropic Chat API wrapper."""
import os

import pytest

from langchain_anthropic.llms import Anthropic

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_model_name_param() -> None:
    llm = Anthropic(model_name="foo")
    assert llm.model == "foo"


def test_model_param() -> None:
    llm = Anthropic(model="foo")
    assert llm.model == "foo"


def test_model_kwargs() -> None:
    llm = Anthropic(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        Anthropic(model_kwargs={"max_tokens_to_sample": 5})


def test_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = Anthropic(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


def test_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    Anthropic(model="test", anthropic_api_key="test")


def test_get_num_tokens() -> None:
    llm = Anthropic(model="test", anthropic_api_key="test")
    assert llm.get_num_tokens("Hello claude") > 0
