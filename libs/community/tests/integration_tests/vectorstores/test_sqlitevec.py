from typing import List, Optional

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import SQLiteVec
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _sqlite_vec_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> SQLiteVec:
    return SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        table="test",
        db_file=":memory:",
    )


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec() -> None:
    """Test end to end construction and search."""
    docsearch = _sqlite_vec_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={})]


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    distances = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert distances[0] < distances[1] < distances[2]


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    docsearch.add_texts(texts, metadatas)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_search_multiple_tables() -> None:
    """Test end to end construction and search with multiple tables."""
    docsearch_1 = SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        table="table_1",
        db_file=":memory:",  ## change to local storage for testing
    )

    docsearch_2 = SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        table="table_2",
        db_file=":memory:",
    )

    output_1 = docsearch_1.similarity_search("foo", k=1)
    output_2 = docsearch_2.similarity_search("foo", k=1)

    assert output_1 == [Document(page_content="foo", metadata={})]
    assert output_2 == [Document(page_content="foo", metadata={})]
