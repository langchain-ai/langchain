import os

import pytest

from langchain_neospace import NeoSpaceEmbeddings

os.environ["NEOSPACE_API_KEY"] = "foo"


def test_neospace_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        NeoSpaceEmbeddings(model_kwargs={"model": "foo"})


def test_neospace_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = NeoSpaceEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}
