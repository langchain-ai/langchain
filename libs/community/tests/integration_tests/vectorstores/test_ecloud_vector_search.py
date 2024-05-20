"""Test EcloudESVectorStore functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document

from langchain_community.vectorstores.ecloud_vector_search import EcloudESVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)

if TYPE_CHECKING:
    from elasticsearch.client import Elasticsearch

user = "elastic"
password = "*****"
ES_URL = "http://localhost:9200"


def _ecloud_vector_db_from_texts(
    metadatas: Optional[List[dict]] = None, index_name: str = "testknn"
) -> EcloudESVectorStore:
    return EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        es_url=ES_URL,
        user=user,
        password=password,
        index_name=index_name,
        refresh_indices=True,
    )


def delete_index(es: Elasticsearch, index: str) -> None:
    """Delete the specific index"""
    try:
        es.indices.delete(index)
    except Exception:
        pass


def test_ecloud_vector_db() -> None:
    """Test end to end construction and search."""
    index_name = "testknn1"
    docsearch = _ecloud_vector_db_from_texts(index_name=index_name)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
    delete_index(docsearch.client, index_name)


def test_ecloud_vector_index_settings() -> None:
    index_name = "testknn2"
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        es_url=ES_URL,
        user=user,
        password=password,
        index_name=index_name,
        refresh_indices=True,
        vector_field="my_vector",
        text_field="custom_text",
        time_out=120,
    )
    res = docsearch.client.indices.get_settings(index=index_name)
    assert res[index_name]["settings"]["index"]["number_of_shards"] == "1"
    assert res[index_name]["settings"]["index"]["number_of_replicas"] == "1"

    delete_index(docsearch.client, index_name)

    index_name = "testknn3"
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        es_url=ES_URL,
        user=user,
        password=password,
        index_name=index_name,
        refresh_indices=True,
        vector_field="my_vector",
        text_field="custom_text",
        index_settings={"index": {"number_of_shards": "3", "number_of_replicas": "0"}},
    )
    res = docsearch.client.indices.get_settings(index=index_name)
    assert res[index_name]["settings"]["index"]["number_of_shards"] == "3"
    assert res[index_name]["settings"]["index"]["number_of_replicas"] == "0"
    delete_index(docsearch.client, index_name)


def test_similarity_search_with_score() -> None:
    """Test similarity search with score using Approximate Search."""
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    index_name = "testknn4"
    docsearch = _ecloud_vector_db_from_texts(metadatas=metadatas, index_name=index_name)
    output = docsearch.similarity_search_with_score("foo", k=2)
    assert output == [
        (Document(page_content="foo", metadata={"page": 0}), 2.0),
        (Document(page_content="bar", metadata={"page": 1}), 1.9486833),
    ]
    delete_index(docsearch.client, index_name)


def test_ecloud_with_custom_field_name() -> None:
    """Test indexing and search using custom vector field and text field name."""
    index_name = "testknn5"
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        es_url=ES_URL,
        user=user,
        password=password,
        index_name=index_name,
        refresh_indices=True,
        vector_field="my_vector",
        text_field="custom_text",
    )
    output = docsearch.similarity_search(
        "foo", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo")]

    text_input = ["test", "add", "text", "method"]
    EcloudESVectorStore.add_texts(
        docsearch, text_input, vector_field="my_vector", text_field="custom_text"
    )
    output = docsearch.similarity_search(
        "add", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo")]
    delete_index(docsearch.client, index_name)


def test_ecloud_with_metadatas() -> None:
    """Test end to end indexing and search with metadata."""
    index_name = "testknn6"
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        index_name=index_name,
        refresh_indices=True,
        metadatas=metadatas,
        es_url=ES_URL,
        user=user,
        password=password,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]
    delete_index(docsearch.client, index_name)


def test_add_text() -> None:
    """Test adding additional text elements to existing index."""
    index_name = "testknn7"
    text_input = ["test", "add", "text", "method"]
    metadatas = [{"page": i} for i in range(len(text_input))]
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        index_name=index_name,
        refresh_indices=True,
        es_url=ES_URL,
        user=user,
        password=password,
    )
    docids = EcloudESVectorStore.add_texts(docsearch, text_input, metadatas)
    assert len(docids) == len(text_input)
    delete_index(docsearch.client, index_name)


def test_dense_float_vector_lsh_cosine() -> None:
    """
    Test indexing with vector type knn_dense_float_vector and
    model-similarity of lsh-cosine
    this mapping is compatible with model of exact and similarity of l2/cosine
    this mapping is compatible with model of lsh and similarity of cosine
    """
    index_name = "testknn9"
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        index_name=index_name,
        refresh_indices=True,
        es_url=ES_URL,
        user=user,
        password=password,
        text_field="my_text",
        vector_field="my_vec",
        vector_type="knn_dense_float_vector",
        vector_params={"model": "lsh", "similarity": "cosine", "L": 99, "k": 1},
    )
    output = docsearch.similarity_search(
        "foo",
        k=1,
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search(
        "foo",
        k=1,
        search_params={
            "model": "exact",
            "similarity": "l2",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search(
        "foo",
        k=1,
        search_params={
            "model": "exact",
            "similarity": "cosine",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search(
        "foo",
        k=1,
        search_params={
            "model": "lsh",
            "similarity": "cosine",
            "candidates": 1,
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    delete_index(docsearch.client, index_name)


def test_dense_float_vector_exact_with_filter() -> None:
    """
    Test indexing with vector type knn_dense_float_vector and
    default model/similarity
    this mapping is compatible with model of exact and
    similarity of l2/cosine
    """
    index_name = "testknn15"
    docsearch = EcloudESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        index_name=index_name,
        refresh_indices=True,
        es_url=ES_URL,
        user=user,
        password=password,
        text_field="my_text",
        vector_field="my_vec",
        vector_type="knn_dense_float_vector",
    )

    output = docsearch.similarity_search(
        "foo",
        k=1,
        filter={"match_all": {}},
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search(
        "bar",
        k=2,
        filter={"term": {"my_text.keyword": "bar"}},
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="bar")]

    output = docsearch.similarity_search(
        "bar",
        k=2,
        filter={"term": {"my_text.keyword": "foo"}},
        search_params={
            "model": "exact",
            "similarity": "l2",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search(
        "foo",
        k=2,
        filter={"bool": {"filter": {"term": {"my_text.keyword": "bar"}}}},
        search_params={
            "model": "exact",
            "similarity": "cosine",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="bar")]

    output = docsearch.similarity_search(
        "foo",
        k=2,
        filter={"bool": {"filter": [{"term": {"my_text.keyword": "bar"}}]}},
        search_params={
            "model": "exact",
            "similarity": "cosine",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    assert output == [Document(page_content="bar")]

    delete_index(docsearch.client, index_name)
