from langchain_community.embeddings.hunyuan import HunyuanEmbeddings


def test_embedding_query() -> None:
    query = "foo"
    embedding = HunyuanEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) == 1024


def test_embedding_document() -> None:
    documents = ["foo bar"]
    embedding = HunyuanEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_embedding_documents() -> None:
    documents = ["foo", "bar"]
    embedding = HunyuanEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
