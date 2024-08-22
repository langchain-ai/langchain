"""Test VDMS functionality."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import VDMS
from langchain_community.vectorstores.vdms import VDMS_Client, embedding2bytes
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

if TYPE_CHECKING:
    import vdms

logging.basicConfig(level=logging.DEBUG)
embedding_function = FakeEmbeddings()


# The connection string matches the default settings in the docker-compose file
# located in the root of the repository: [root]/docker/docker-compose.yml
# To spin up a detached VDMS server:
# cd [root]/docker
# docker compose up -d vdms
@pytest.fixture
@pytest.mark.enable_socket
def vdms_client() -> vdms.vdms:
    return VDMS_Client(
        host=os.getenv("VDMS_DBHOST", "localhost"),
        port=int(os.getenv("VDMS_DBPORT", 6025)),
    )


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_init_from_client(vdms_client: vdms.vdms) -> None:
    _ = VDMS(  # type: ignore[call-arg]
        embedding=embedding_function,
        client=vdms_client,
    )


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_from_texts_with_metadatas(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and search."""
    collection_name = "test_from_texts_with_metadatas"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_from_texts_with_metadatas_{i}" for i in range(len(texts))]
    metadatas = [{"page": str(i)} for i in range(1, len(texts) + 1)]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [
        Document(page_content="foo", metadata={"page": "1", "id": ids[0]})
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_from_texts_with_metadatas_with_scores(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and scored search."""
    collection_name = "test_from_texts_with_metadatas_with_scores"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_from_texts_with_metadatas_with_scores_{i}" for i in range(len(texts))]
    metadatas = [{"page": str(i)} for i in range(1, len(texts) + 1)]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, fetch_k=1)
    assert output == [
        (Document(page_content="foo", metadata={"page": "1", "id": ids[0]}), 0.0)
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_from_texts_with_metadatas_with_scores_using_vector(
    vdms_client: vdms.vdms,
) -> None:
    """Test end to end construction and scored search, using embedding vector."""
    collection_name = "test_from_texts_with_metadatas_with_scores_using_vector"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_from_texts_with_metadatas_{i}" for i in range(len(texts))]
    metadatas = [{"page": str(i)} for i in range(1, len(texts) + 1)]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch._similarity_search_with_relevance_scores("foo", k=1)
    assert output == [
        (Document(page_content="foo", metadata={"page": "1", "id": ids[0]}), 0.0)
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_search_filter(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and search with metadata filtering."""
    collection_name = "test_search_filter"
    texts = ["far", "bar", "baz"]
    ids = [f"test_search_filter_{i}" for i in range(len(texts))]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch.similarity_search(
        "far", k=1, filter={"first_letter": ["==", "f"]}
    )
    assert output == [
        Document(page_content="far", metadata={"first_letter": "f", "id": ids[0]})
    ]
    output = docsearch.similarity_search(
        "far", k=2, filter={"first_letter": ["==", "b"]}
    )
    assert output == [
        Document(page_content="bar", metadata={"first_letter": "b", "id": ids[1]}),
        Document(page_content="baz", metadata={"first_letter": "b", "id": ids[2]}),
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_search_filter_with_scores(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and scored search with metadata filtering."""
    collection_name = "test_search_filter_with_scores"
    texts = ["far", "bar", "baz"]
    ids = [f"test_search_filter_with_scores_{i}" for i in range(len(texts))]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch.similarity_search_with_score(
        "far", k=1, filter={"first_letter": ["==", "f"]}
    )
    assert output == [
        (
            Document(page_content="far", metadata={"first_letter": "f", "id": ids[0]}),
            0.0,
        )
    ]

    output = docsearch.similarity_search_with_score(
        "far", k=2, filter={"first_letter": ["==", "b"]}
    )
    assert output == [
        (
            Document(page_content="bar", metadata={"first_letter": "b", "id": ids[1]}),
            1.0,
        ),
        (
            Document(page_content="baz", metadata={"first_letter": "b", "id": ids[2]}),
            4.0,
        ),
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_mmr(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and search."""
    collection_name = "test_mmr"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_mmr_{i}" for i in range(len(texts))]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"id": ids[0]})]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_mmr_by_vector(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and search."""
    collection_name = "test_mmr_by_vector"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_mmr_by_vector_{i}" for i in range(len(texts))]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        collection_name=collection_name,
        client=vdms_client,
    )
    embedded_query = embedding_function.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo", metadata={"id": ids[0]})]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_with_include_parameter(vdms_client: vdms.vdms) -> None:
    """Test end to end construction and include parameter."""
    collection_name = "test_with_include_parameter"
    texts = ["foo", "bar", "baz"]
    docsearch = VDMS.from_texts(
        texts=texts,
        embedding=embedding_function,
        collection_name=collection_name,
        client=vdms_client,
    )

    response, response_array = docsearch.get(collection_name, include=["embeddings"])
    for emb in embedding_function.embed_documents(texts):
        assert embedding2bytes(emb) in response_array

    response, response_array = docsearch.get(collection_name)
    assert response_array == []


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_update_document(vdms_client: vdms.vdms) -> None:
    """Test the update_document function in the VDMS class."""
    collection_name = "test_update_document"

    # Make a consistent embedding
    const_embedding_function = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "1"})

    # Initialize a VDMS instance with the original document
    docsearch = VDMS.from_documents(
        client=vdms_client,
        collection_name=collection_name,
        documents=[original_doc],
        embedding=const_embedding_function,
        ids=[document_id],
    )
    old_response, old_embedding = docsearch.get(
        collection_name,
        constraints={"id": ["==", document_id]},
        include=["metadata", "embeddings"],
    )
    # old_embedding = response_array[0]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(page_content=updated_content, metadata={"page": "1"})

    # Update the document in the VDMS instance
    docsearch.update_document(
        collection_name, document_id=document_id, document=updated_doc
    )

    # Perform a similarity search with the updated content
    output = docsearch.similarity_search(updated_content, k=3)[0]

    # Assert that the updated document is returned by the search
    assert output == Document(
        page_content=updated_content, metadata={"page": "1", "id": document_id}
    )

    # Assert that the new embedding is correct
    new_response, new_embedding = docsearch.get(
        collection_name,
        constraints={"id": ["==", document_id]},
        include=["metadata", "embeddings"],
    )
    # new_embedding = response_array[0]

    assert new_embedding[0] == embedding2bytes(
        const_embedding_function.embed_documents([updated_content])[0]
    )
    assert new_embedding != old_embedding

    assert (
        new_response[0]["FindDescriptor"]["entities"][0]["content"]
        != old_response[0]["FindDescriptor"]["entities"][0]["content"]
    )


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_with_relevance_score(vdms_client: vdms.vdms) -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    collection_name = "test_with_relevance_score"
    texts = ["foo", "bar", "baz"]
    ids = [f"test_relevance_scores_{i}" for i in range(len(texts))]
    metadatas = [{"page": str(i)} for i in range(1, len(texts) + 1)]
    docsearch = VDMS.from_texts(
        texts=texts,
        ids=ids,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=collection_name,
        client=vdms_client,
    )
    output = docsearch._similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "1", "id": ids[0]}), 0.0),
        (Document(page_content="bar", metadata={"page": "2", "id": ids[1]}), 0.25),
        (Document(page_content="baz", metadata={"page": "3", "id": ids[2]}), 1.0),
    ]


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_add_documents_no_metadata(vdms_client: vdms.vdms) -> None:
    collection_name = "test_add_documents_no_metadata"
    db = VDMS(  # type: ignore[call-arg]
        collection_name=collection_name,
        embedding=embedding_function,
        client=vdms_client,
    )
    db.add_documents([Document(page_content="foo")])


@pytest.mark.requires("vdms")
@pytest.mark.enable_socket
def test_add_documents_mixed_metadata(vdms_client: vdms.vdms) -> None:
    collection_name = "test_add_documents_mixed_metadata"
    db = VDMS(  # type: ignore[call-arg]
        collection_name=collection_name,
        embedding=embedding_function,
        client=vdms_client,
    )

    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["10", "11"]
    actual_ids = db.add_documents(docs, ids=ids)
    assert actual_ids == ids

    search = db.similarity_search("foo bar", k=2)
    docs[0].metadata = {"id": ids[0]}
    docs[1].metadata["id"] = ids[1]
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )
