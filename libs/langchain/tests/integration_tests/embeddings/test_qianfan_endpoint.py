"""Test Baidu Qianfan Embedding Endpoint."""
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint


def test_embedding_multiple_documents() -> None:
    documents = ["foo", "bar"]
    embedding = QianfanEmbeddingsEndpoint()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384


def test_embedding_query() -> None:
    query = "foo"
    embedding = QianfanEmbeddingsEndpoint()
    output = embedding.embed_query(query)
    assert len(output) == 384


def test_model() -> None:
    documents = ["hi", "qianfan"]
    embedding = QianfanEmbeddingsEndpoint(model="Embedding-V1")
    output = embedding.embed_documents(documents)
    assert len(output) == 2


def test_init_kwargs() -> None:
    chat = QianfanEmbeddingsEndpoint(qianfan_ak="11", qianfan_sk="22", init_kwargs={"query_per_second": 12})
    assert chat.client._client._rate_limiter._sync_limiter._query_per_period == 12
