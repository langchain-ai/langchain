"""Test OceanBase functionality."""

import os

import pytest
import sqlalchemy
from langchain_core.documents import Document

from langchain_community.vectorstores.oceanbase import (
    OceanBase
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from tests.integration_tests.vectorstores.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
)

CONNECTION_ARGS = {
    "host": os.environ.get("TEST_OCEANBASE_HOST", "localhost"),
    "port": os.environ.get("TEST_OCEANBASE_PORT", "2881"),
    "user": os.environ.get("TEST_OCEANBASE_USER", "root@test"),
    "password": os.environ.get("TEST_OCEANBASE_PWD", ""),
    "db_name": os.environ.get("TEST_OCEANBASE_DBNAME", "test")
}

def test_oceanbase() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = OceanBase.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        connection_args=CONNECTION_ARGS,
        drop_old=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    print(f"output={output}")
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
    )
    output = docsearch.similarity_search_with_score("foo", k=1, fltr="metadata->'$.page' = '0'")
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]

# def test_oceanbase_with_filter_distant_match() -> None:
#     """Test end to end construction and search."""
#     texts = ["foo", "bar", "baz"]
#     metadatas = [{"page": str(i)} for i in range(len(texts))]
#     docsearch = OceanBase.from_texts(
#         texts=texts,
#         embedding=FakeEmbeddings(),
#         metadatas=metadatas,
#         connection_args=CONNECTION_ARGS,
#         drop_old=True,
#     )
#     output = docsearch.similarity_search_with_score("foo", k=1, fltr="metadata->'$.page' = '2'")
#     assert output == [
#         (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)
#     ]