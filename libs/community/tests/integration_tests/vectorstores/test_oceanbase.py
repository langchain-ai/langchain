"""Test OceanBase functionality."""

import logging
import os

from langchain_core.documents import Document

from langchain_community.vectorstores.oceanbase import OceanBase
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CONNECTION_ARGS = {
    "host": os.environ.get("TEST_OCEANBASE_HOST", "localhost"),
    "port": os.environ.get("TEST_OCEANBASE_PORT", "2881"),
    "user": os.environ.get("TEST_OCEANBASE_USER", "root@test"),
    "password": os.environ.get("TEST_OCEANBASE_PWD", ""),
    "db_name": os.environ.get("TEST_OCEANBASE_DBNAME", "test"),
}


def test_oceanbase() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_oceanbase_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_oceanbase_with_metadatas_with_scores() -> None:
    """Test end to end construction and search with scores."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_oceanbase_with_filter_match() -> None:
    """Test end to end construction and search with filter."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=1, fltr="metadata->'$.page' = '0'"
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_oceanbase_with_filter_no_match() -> None:
    """Test end to end construction and search in case of mismatches."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=1, fltr="metadata->'$.page' = '5'"
    )
    assert output == []


def test_oceanbase_delete_docs() -> None:
    """Test docs deletion."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        ids=["1", "2", "3"],
        drop_old=True,
        normalize=True,
    )
    docsearch.delete(ids=["1", "2"])
    res = docsearch.obvector.perform_raw_text_sql("SELECT id FROM langchain_vector")
    assert [r[0] for r in res] == ["3"]


def test_pgvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        normalize=True,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7},
    )
    output = retriever.invoke("summer")
    assert output == [
        Document(metadata={"page": "0"}, page_content="foo"),
        Document(metadata={"page": "1"}, page_content="bar"),
    ]
