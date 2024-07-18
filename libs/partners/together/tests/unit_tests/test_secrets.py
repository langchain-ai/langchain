from langchain_together import ChatTogether, TogetherEmbeddings


def test_chat_together_secrets() -> None:
    o = ChatTogether(together_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_together_embeddings_secrets() -> None:
    o = TogetherEmbeddings(together_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
