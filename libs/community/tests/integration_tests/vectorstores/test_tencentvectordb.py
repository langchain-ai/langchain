"""Test TencentVectorDB functionality."""

import time
from typing import List, Optional

from langchain_core.documents import Document

from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import ConnectionParams
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _tencent_vector_db_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> TencentVectorDB:
    conn_params = ConnectionParams(
        url="http://10.0.X.X",
        key="eC4bLRy2va******************************",
        username="root",
        timeout=20,
    )
    return TencentVectorDB.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        connection_params=conn_params,
        drop_old=drop,
    )


def test_tencent_vector_db() -> None:
    """Test end to end construction and search."""
    docsearch = _tencent_vector_db_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_tencent_vector_db_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _tencent_vector_db_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]


def test_tencent_vector_db_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _tencent_vector_db_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]


def test_tencent_vector_db_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _tencent_vector_db_from_texts(metadatas=metadatas)
    docsearch.add_texts(texts, metadatas)
    time.sleep(3)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_tencent_vector_db_no_drop() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _tencent_vector_db_from_texts(metadatas=metadatas)
    del docsearch
    docsearch = _tencent_vector_db_from_texts(metadatas=metadatas, drop=False)
    time.sleep(3)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6
