"""Test Milvus functionality."""
from typing import List, Optional

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import Milvus
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _milvus_from_texts(metadatas: Optional[List[dict]] = None) -> Milvus:
    return Milvus.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )


def _milvus_from_text_with_index(metadatas: Optional[List[dict]] = None, index_type: str = 'IVF_FLAT') -> Milvus:
    return Milvus.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        index_type=index_type
    )


def test_milvus() -> None:
    """Test end to end construction and search."""
    docsearch = _milvus_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_milvus_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] < scores[1] < scores[2]


def test_milvus_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="baz", metadata={"page": 2}),
    ]


def test_milvus_IVF_FLAT() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_text_with_index(metadatas=metadatas, index_type='IVF_FLAT')
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_milvus_IVF_SQ8() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_text_with_index(metadatas=metadatas, index_type='IVF_SQ8')
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_milvus_IVF_PQ() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_text_with_index(metadatas=metadatas, index_type='IVF_PQ')
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_milvus_HNSW() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_text_with_index(metadatas=metadatas, index_type='HNSW')
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_milvus_ANNOY() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_text_with_index(metadatas=metadatas, index_type='ANNOY')
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_milvus_unsupported_index() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    with pytest.raises(AssertionError):
        _milvus_from_text_with_index(metadatas=metadatas, index_type='UNSUPPORTED_INDEX')
