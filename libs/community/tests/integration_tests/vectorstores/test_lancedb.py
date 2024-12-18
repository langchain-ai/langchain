from typing import Any

import pytest

from langchain_community.vectorstores import LanceDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def import_lancedb() -> Any:
    try:
        import lancedb
    except ImportError as e:
        raise ImportError(
            "Could not import lancedb package. "
            "Please install it with `pip install lancedb`."
        ) from e
    return lancedb


@pytest.mark.requires("lancedb")
def test_lancedb_with_connection() -> None:
    lancedb = import_lancedb()

    embeddings = FakeEmbeddings()
    db = lancedb.connect("/tmp/lancedb_connection")
    texts = ["text 1", "text 2", "item 3"]
    store = LanceDB(connection=db, embedding=embeddings)
    store.add_texts(texts)

    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

    store.delete(filter="text = 'text 1'")
    assert store.get_table().count_rows() == 2


@pytest.mark.requires("lancedb")
def test_lancedb_without_connection() -> None:
    embeddings = FakeEmbeddings()
    texts = ["text 1", "text 2", "item 3"]

    store = LanceDB(embedding=embeddings)
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


@pytest.mark.requires("lancedb")
def test_lancedb_add_texts() -> None:
    embeddings = FakeEmbeddings()

    store = LanceDB(embedding=embeddings)
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts


@pytest.mark.requires("lancedb")
def test_mmr() -> None:
    embeddings = FakeEmbeddings()
    store = LanceDB(embedding=embeddings)
    store.add_texts(["text 1", "text 2", "item 3"])
    result = store.max_marginal_relevance_search(query="text")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

    result = store.max_marginal_relevance_search_by_vector(
        embeddings.embed_query("text")
    )
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


@pytest.mark.requires("lancedb")
def test_lancedb_delete() -> None:
    embeddings = FakeEmbeddings()

    store = LanceDB(embedding=embeddings)
    store.add_texts(["text 1", "text 2", "item 3"])
    store.delete(filter="text = 'text 1'")
    assert store.get_table().count_rows() == 2


@pytest.mark.requires("lancedb")
def test_lancedb_delete_by_ids() -> None:
    embeddings = FakeEmbeddings()

    store = LanceDB(embedding=embeddings, id_key="pk")
    ids = store.add_texts(["text 1", "text 2", "item 3"])
    store.delete(ids=ids)
    assert store.get_table().count_rows() == 0


@pytest.mark.requires("lancedb")
def test_lancedb_all_searches() -> None:
    embeddings = FakeEmbeddings()

    store = LanceDB(embedding=embeddings)
    store.add_texts(["text 1", "text 2", "item 3"])
    result_1 = store.similarity_search_with_relevance_scores(
        "text 1", distance="cosine"
    )
    assert len(result_1[0]) == 2
    assert "text 1" in result_1[0][0].page_content

    result_2 = store.similarity_search_by_vector(embeddings.embed_query("text 1"))
    assert "text 1" in result_2[0].page_content

    result_3 = store.similarity_search_by_vector_with_relevance_scores(
        embeddings.embed_query("text 1")
    )
    assert len(result_3[0]) == 2  # type: ignore
    assert "text 1" in result_3[0][0].page_content  # type: ignore


@pytest.mark.requires("lancedb")
def test_lancedb_no_metadata() -> None:
    lancedb = import_lancedb()
    embeddings = FakeEmbeddings()
    # Connect to a temporary LanceDB instance
    db = lancedb.connect("/tmp/lancedb_no_metadata_test")
    # Create data without the 'metadata' field
    texts = ["text 1", "text 2", "item 3"]
    data = []
    for idx, text in enumerate(texts):
        embedding = embeddings.embed_documents([text])[0]
        data.append(
            {
                "vector": embedding,
                "id": str(idx),
                "text": text,
                # Note: We're deliberately not including 'metadata' here
            }
        )
    # Create the table without 'metadata' column
    db.create_table("vectorstore_no_metadata", data=data)
    # Initialize LanceDB with the existing connection and table name
    store = LanceDB(
        connection=db,
        embedding=embeddings,
        table_name="vectorstore_no_metadata",
    )
    # Perform a similarity search
    result = store.similarity_search("text 1")
    # Verify that the metadata in the Document objects is an empty dictionary
    for doc in result:
        assert (
            doc.metadata == {}
        ), "Expected empty metadata when 'metadata' column is missing"
    # Clean up by deleting the table (optional)
    db.drop_table("vectorstore_no_metadata")
