"""Test embedding model integration."""

import os

import pytest  # type: ignore[import-not-found]

from langchain_together import TogetherEmbeddings

os.environ["TOGETHER_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    TogetherEmbeddings()


def test_together_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        TogetherEmbeddings(model_kwargs={"model": "foo"})


def test_together_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = TogetherEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}
