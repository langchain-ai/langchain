"""Test Vald functionality."""
import time
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import Vald
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)

"""
To run, you should have a Vald cluster.
https://github.com/vdaas/vald/blob/main/docs/tutorial/get-started.md
"""

WAIT_TIME = 90


def _vald_from_texts(
    metadatas: Optional[List[dict]] = None,
    host: str = "localhost",
    port: int = 8080,
    skip_strict_exist_check: bool = True,
) -> Vald:
    return Vald.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        host=host,
        port=port,
        skip_strict_exist_check=skip_strict_exist_check,
    )


def test_vald_add_texts() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)  # Wait for CreateIndex

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 3

    texts = ["a", "b", "c"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch.add_texts(texts, metadatas)
    time.sleep(WAIT_TIME)  # Wait for CreateIndex

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_vald_delete() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 3

    docsearch.delete(["foo"])
    time.sleep(WAIT_TIME)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 2


def test_vald_search() -> None:
    """Test end to end construction and search."""
    docsearch = _vald_from_texts()
    time.sleep(WAIT_TIME)

    output = docsearch.similarity_search("foo", k=3)

    assert output == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]


def test_vald_search_with_score() -> None:
    """Test end to end construction and search with scores."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]

    assert docs == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert scores[0] < scores[1] < scores[2]


def test_vald_search_by_vector() -> None:
    """Test end to end construction and search by vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    embedding = FakeEmbeddings().embed_query("foo")
    output = docsearch.similarity_search_by_vector(embedding, k=3)

    assert output == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]


def test_vald_search_with_score_by_vector() -> None:
    """Test end to end construction and search with scores by vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    embedding = FakeEmbeddings().embed_query("foo")
    output = docsearch.similarity_search_with_score_by_vector(embedding, k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]

    assert docs == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert scores[0] < scores[1] < scores[2]


def test_vald_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)

    assert output == [
        Document(page_content="foo"),
        Document(page_content="bar"),
    ]


def test_vald_max_marginal_relevance_search_by_vector() -> None:
    """Test end to end construction and MRR search by vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vald_from_texts(metadatas=metadatas)
    time.sleep(WAIT_TIME)

    embedding = FakeEmbeddings().embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(
        embedding, k=2, fetch_k=3
    )

    assert output == [
        Document(page_content="foo"),
        Document(page_content="bar"),
    ]
