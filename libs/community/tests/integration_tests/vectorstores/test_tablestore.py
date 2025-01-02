"""Test tablestore functionality."""

import os

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.tablestore import TablestoreVectorStore


def test_tablestore() -> None:
    """Test end to end construction and search."""
    test_embedding_dimension_size = 4
    embeddings = FakeEmbeddings(size=test_embedding_dimension_size)

    end_point = os.getenv("end_point")
    instance_name = os.getenv("instance_name")
    access_key_id = os.getenv("access_key_id")
    access_key_secret = os.getenv("access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )
    """
        1. create vector store 
    """
    store = TablestoreVectorStore(
        embedding=embeddings,
        endpoint=end_point,
        instance_name=instance_name,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        vector_dimension=test_embedding_dimension_size,
    )

    """
        2. create table and index. (only needs to be run once)
    """
    store.create_table_if_not_exist()
    store.create_search_index_if_not_exist()

    """
        3. add document
    """
    store.add_documents(
        [
            Document(
                id="1",
                page_content="1 hello world",
                metadata={"type": "pc", "time": 2000},
            ),
            Document(
                id="2", page_content="abc world", metadata={"type": "pc", "time": 2009}
            ),
            Document(
                id="3",
                page_content="3 text world",
                metadata={"type": "sky", "time": 2010},
            ),
            Document(
                id="4", page_content="hi world", metadata={"type": "sky", "time": 2030}
            ),
            Document(
                id="5", page_content="hi world", metadata={"type": "sky", "time": 2030}
            ),
        ]
    )

    """
        4. delete document
    """
    assert store.delete(["3"])

    """
        5. get document
    """
    get_docs = store.get_by_ids(["1", "4"])
    assert len(get_docs) == 2
    assert get_docs[0].id == "1"
    assert get_docs[1].id == "4"

    """
        6. similarity_search
    """
    search_result = store.similarity_search_with_score(query="hello world", k=2)
    assert len(search_result) == 2


def test_tablestore_add_documents() -> None:
    embeddings = FakeEmbeddings(size=128)
    store = TablestoreVectorStore(
        embedding=embeddings,
        endpoint="http://test.a.com",
        instance_name="test",
        access_key_id="test",
        access_key_secret="test",
        vector_dimension=512,
    )
    doc = Document(
        id="1",
        page_content="1 hello world",
        metadata={"type": "pc", "time": 2000},
    )

    try:
        store.add_documents([doc])
        raise RuntimeError("should failed")
    except Exception as e:
        assert "not the same as" in e.args[0]
