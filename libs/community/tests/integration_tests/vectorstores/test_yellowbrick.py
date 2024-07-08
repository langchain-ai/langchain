import logging
from typing import List, Optional

import pytest

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Yellowbrick
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    fake_texts,
)

YELLOWBRICK_URL = "postgres://username:password@host:port/database"
YELLOWBRICK_TABLE = "test_table"
YELLOWBRICK_CONTENT = "test_table_content"
YELLOWBRICK_SCHEMA = "test_schema"


def _yellowbrick_vector_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Yellowbrick:
    db = Yellowbrick.from_texts(
        fake_texts,
        ConsistentFakeEmbeddings(),
        metadatas,
        YELLOWBRICK_URL,
        table=YELLOWBRICK_TABLE,
        schema=YELLOWBRICK_SCHEMA,
        drop=drop,
    )
    db.logger.setLevel(logging.DEBUG)
    return db


def _yellowbrick_vector_from_texts_no_schema(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Yellowbrick:
    db = Yellowbrick.from_texts(
        fake_texts,
        ConsistentFakeEmbeddings(),
        metadatas,
        YELLOWBRICK_URL,
        table=YELLOWBRICK_TABLE,
        drop=drop,
    )
    db.logger.setLevel(logging.DEBUG)
    return db


@pytest.mark.requires("yb-vss")
def test_yellowbrick() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_text() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_delete() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        added_docs = docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(added_docs)
        output = docsearch.similarity_search("oof", k=1)
        assert output != [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_delete_all() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(delete_all=True)
        output = docsearch.similarity_search("oof", k=1)
        assert output != [Document(page_content="oof", metadata={})]
        output = docsearch.similarity_search("foo", k=1)
        assert output != [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search_update() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_delete() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        added_docs = docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(added_docs)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_delete_all() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(delete_all=True)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output != [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
        output = docsearch.similarity_search_with_score("foo", k=3)
        docs = [o[0] for o in output]
        distances = [o[1] for o in output]
        assert docs == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
            Document(page_content="baz", metadata={"page": 2}),
        ]
        assert distances[0] > distances[1] > distances[2]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_extra() -> None:
    """Test end to end construction and MRR search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
        docsearch.add_texts(texts, metadatas)
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 6
