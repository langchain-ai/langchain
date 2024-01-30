from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)


def test_chat_openai_secrets() -> None:
    o = ChatOpenAI(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_openai_secrets() -> None:
    o = OpenAI(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_openai_embeddings_secrets() -> None:
    o = OpenAIEmbeddings(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_azure_chat_openai_secrets() -> None:
    o = AzureChatOpenAI(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s


def test_azure_openai_secrets() -> None:
    o = AzureOpenAI(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s


def test_azure_openai_embeddings_secrets() -> None:
    o = AzureOpenAIEmbeddings(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s
