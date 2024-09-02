"""Test Baidu Qianfan Embedding Endpoint."""

from typing import cast

from langchain_core.pydantic_v1 import SecretStr

from langchain_community.embeddings.baidu_qianfan_endpoint import (
    QianfanEmbeddingsEndpoint,
)


def test_embedding_multiple_documents() -> None:
    documents = ["foo", "bar"]
    embedding = QianfanEmbeddingsEndpoint()  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384


def test_embedding_query() -> None:
    query = "foo"
    embedding = QianfanEmbeddingsEndpoint()  # type: ignore[call-arg]
    output = embedding.embed_query(query)
    assert len(output) == 384


def test_model() -> None:
    documents = ["hi", "qianfan"]
    embedding = QianfanEmbeddingsEndpoint(model="Embedding-V1")  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2


def test_rate_limit() -> None:
    llm = QianfanEmbeddingsEndpoint(  # type: ignore[call-arg]
        model="Embedding-V1", init_kwargs={"query_per_second": 2}
    )
    assert llm.client._client._rate_limiter._sync_limiter._query_per_second == 2
    documents = ["foo", "bar"]
    output = llm.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384


def test_initialization_with_alias() -> None:
    """Test qianfan embedding model initialization with alias."""
    api_key = "your-api-key"
    secret_key = "your-secret-key"

    embeddings = QianfanEmbeddingsEndpoint(  # type: ignore[arg-type, call-arg]
        api_key=api_key,  # type: ignore[arg-type]
        secret_key=secret_key,  # type: ignore[arg-type]
    )

    assert cast(SecretStr, embeddings.qianfan_ak).get_secret_value() == api_key
    assert cast(SecretStr, embeddings.qianfan_sk).get_secret_value() == secret_key
