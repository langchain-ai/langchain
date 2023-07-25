import os

import pytest

from langchain.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    llm = OpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo")
    assert llm.model_name == "foo"


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"model_name": "foo"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAI(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}
