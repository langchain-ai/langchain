"""Test OpenAI API wrapper."""

import pytest

from langchain.llms.openai import OpenAI


def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI(max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_extra_kwargs() -> None:
    """Test extra kwargs to openai."""
    # Check that foo is saved in extra_kwargs.
    llm = OpenAI(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = OpenAI(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        OpenAI(foo=3, model_kwargs={"foo": 2})


def test_openai_stop_valid() -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(stop="3", temperature=0)
    first_output = first_llm(query)
    second_llm = OpenAI(temperature=0)
    second_output = second_llm(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


def test_openai_stop_error() -> None:
    """Test openai stop logic on bad configuration."""
    llm = OpenAI(stop="3", temperature=0)
    with pytest.raises(ValueError):
        llm("write an ordered list of five items", stop=["\n"])
