import pytest

from langchain_community.embeddings import LocalAIEmbeddings


@pytest.mark.requires("openai")
def test_localai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        LocalAIEmbeddings(model_kwargs={"model": "foo"})


@pytest.mark.requires("openai")
def test_localai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = LocalAIEmbeddings(foo="bar", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
def test_localai_base_url() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="random-string", openai_api_base="http://localhost:8080"
    )
    assert llm.openai_api_base == "http://localhost:8080"


@pytest.mark.requires("openai")
def test_localai_client() -> None:
    import openai

    llm = LocalAIEmbeddings(
        openai_api_key="random-string",
        openai_api_base="http://localhost:8080",
        client=(myclient := openai.OpenAI(api_key="foo")),
        async_client=(myasync := openai.AsyncClient(api_key="foo")),
    )
    assert llm.openai_api_base == "http://localhost:8080"
    assert llm.client == myclient
    assert llm.async_client == myasync


@pytest.mark.requires("openai")
def test_localai_proxy() -> None:
    import openai

    llm = LocalAIEmbeddings(
        openai_api_key="random-string",
        openai_api_base="http://localhost:8080",
        openai_proxy="http://localhost:6666",
    )
    assert llm.openai_api_base == "http://localhost:8080"
    assert llm.openai_proxy == "http://localhost:6666"
    with pytest.raises(ValueError):
        LocalAIEmbeddings(
            openai_api_key="random-string",
            openai_api_base="http://localhost:8080",
            client=openai.OpenAI(api_key="foo"),
            openai_proxy="http://localhost:6666",
        )
    with pytest.raises(ValueError):
        LocalAIEmbeddings(
            openai_api_key="random-string",
            openai_api_base="http://localhost:8080",
            async_client=openai.AsyncClient(api_key="foo"),
            openai_proxy="http://localhost:6666",
        )
