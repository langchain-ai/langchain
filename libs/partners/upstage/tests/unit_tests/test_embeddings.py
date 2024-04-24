"""Test embedding model integration."""
import os

import pytest

from langchain_upstage import UpstageEmbeddings

os.environ["UPSTAGE_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    UpstageEmbeddings()


def test_upstage_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        UpstageEmbeddings(model_kwargs={"model": "foo"})


def test_upstage_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = UpstageEmbeddings(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}
