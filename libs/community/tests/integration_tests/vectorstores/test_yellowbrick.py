from typing import List, Optional

import pytest

from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    fake_texts,
)

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Yellowbrick

YELLOWBRICK_URL = "postgres://username:password@host:port/database"
YELLOWBRICK_TABLE = "test_table"


def _yellowbrick_vector_from_texts(
    metadatas: Optional[List[dict]] = None, use_lsh: bool = False, drop: bool = True
) -> Yellowbrick:
    return Yellowbrick.from_texts(
        fake_texts,
        ConsistentFakeEmbeddings(),
        metadatas,
        YELLOWBRICK_URL,
        YELLOWBRICK_TABLE,
        use_lsh=use_lsh,
        drop=drop,
    )


@pytest.mark.requires("yb-vss")
def test_yellowbrick() -> None:
    """Test end to end construction and search."""
    docsearch = _yellowbrick_vector_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert output == [Document(page_content="foo", metadata={})]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_text() -> None:
    """Test end to end construction and search."""
    docsearch = _yellowbrick_vector_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={})]
    texts = ["oof"]
    docsearch.add_texts(texts)
    output = docsearch.similarity_search("oof", k=1)
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert output == [Document(page_content="oof", metadata={})]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search() -> None:
    """Test end to end construction and search."""
    docsearch = _yellowbrick_vector_from_texts()
    docsearch.drop_lsh_index()
    docsearch.create_lsh_index(10)
    docsearch.use_lsh = True
    output = docsearch.similarity_search("foo", k=1)
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert output == [Document(page_content="foo", metadata={})]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search_update() -> None:
    """Test end to end construction and search."""
    docsearch = _yellowbrick_vector_from_texts()
    docsearch.drop_lsh_index()
    docsearch.create_lsh_index(10)
    docsearch.use_lsh = True
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={})]
    texts = ["oof"]
    docsearch.add_texts(texts)
    output = docsearch.similarity_search("oof", k=1)
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert output == [Document(page_content="oof", metadata={})]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    distances = [o[1] for o in output]
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert distances[0] > distances[1] > distances[2]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
    docsearch.add_texts(texts, metadatas)
    output = docsearch.similarity_search("foo", k=10)
    docsearch._drop_table(YELLOWBRICK_TABLE)
    assert len(output) == 6
