import os

import pytest

from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}
