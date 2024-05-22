from langchain_upstage import ChatUpstage, UpstageEmbeddings


def test_chat_upstage_secrets() -> None:
    o = ChatUpstage(upstage_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_upstage_embeddings_secrets() -> None:
    o = UpstageEmbeddings(model="solar-embedding-1-large", upstage_api_key="foo")
    s = str(o)
    assert "foo" not in s
