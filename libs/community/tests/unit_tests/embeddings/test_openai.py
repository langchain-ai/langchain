import pytest

from langchain_community.embeddings.openai import OpenAIEmbeddings


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}
