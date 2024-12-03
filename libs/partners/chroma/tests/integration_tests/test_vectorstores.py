"""Test Chroma functionality."""

import uuid
from typing import (
    AsyncGenerator,
    Generator,
    cast,
)

import chromadb
import pytest  # type: ignore[import-not-found]
import requests
from chromadb.api.client import SharedSystemClient
from chromadb.api.types import Embeddable
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings as Fak

from langchain_chroma.vectorstores import Chroma
from tests.integration_tests.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


class MyEmbeddingFunction:
    def __init__(self, fak: Fak):
        self.fak = fak

    def __call__(self, input: Embeddable) -> list[list[float]]:
        texts = cast(list[str], input)
        return self.fak.embed_documents(texts=texts)


def is_api_accessible(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture()
def client() -> Generator[chromadb.ClientAPI, None, None]:
    SharedSystemClient.clear_system_cache()
    client = chromadb.Client(chromadb.config.Settings())
    yield client


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.fixture()
async def aclient() -> AsyncGenerator[chromadb.AsyncClientAPI, None]:
    SharedSystemClient.clear_system_cache()
    client = await chromadb.AsyncHttpClient()
    yield client


def test_chroma() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    await docsearch.adelete_collection()
    assert output == [Document(page_content="foo")]


def test_chroma_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_with_metadatas_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    await docsearch.adelete_collection()
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_chroma_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.delete_collection()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_with_metadatas_with_scores_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1)
    await docsearch.adelete_collection()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_chroma_with_metadatas_with_vectors() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    vec_1 = embeddings.embed_query(texts[0])
    output = docsearch.similarity_search_with_vectors("foo", k=1)
    docsearch.delete_collection()
    assert output[0][0] == Document(page_content="foo", metadata={"page": "0"})
    assert (output[0][1] == vec_1).all()


def test_chroma_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.similarity_search_by_vector_with_relevance_scores(
        embedding=embedded_query, k=1
    )
    docsearch.delete_collection()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_with_metadatas_with_scores_using_vector_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    embedded_query = embeddings.embed_query("foo")
    output = await docsearch.asimilarity_search_by_vector_with_relevance_scores(
        embedding=embedded_query, k=1
    )
    await docsearch.adelete_collection()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_chroma_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output1 = docsearch.similarity_search("far", k=1, filter={"first_letter": "f"})
    output2 = docsearch.similarity_search("far", k=1, filter={"first_letter": "b"})
    docsearch.delete_collection()
    assert output1 == [Document(page_content="far", metadata={"first_letter": "f"})]
    assert output2 == [Document(page_content="bar", metadata={"first_letter": "b"})]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_search_filter_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output1 = await docsearch.asimilarity_search(
        "far", k=1, filter={"first_letter": "f"}
    )
    output2 = await docsearch.asimilarity_search(
        "far", k=1, filter={"first_letter": "b"}
    )
    await docsearch.adelete_collection()
    assert output1 == [Document(page_content="far", metadata={"first_letter": "f"})]
    assert output2 == [Document(page_content="bar", metadata={"first_letter": "b"})]


def test_chroma_search_filter_with_scores() -> None:
    """Test end to end construction and scored search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output1 = docsearch.similarity_search_with_score(
        "far", k=1, filter={"first_letter": "f"}
    )
    output2 = docsearch.similarity_search_with_score(
        "far", k=1, filter={"first_letter": "b"}
    )
    docsearch.delete_collection()
    assert output1 == [
        (Document(page_content="far", metadata={"first_letter": "f"}), 0.0)
    ]
    assert output2 == [
        (Document(page_content="bar", metadata={"first_letter": "b"}), 1.0)
    ]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_search_filter_with_scores_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Test end to end construction and scored search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output1 = await docsearch.asimilarity_search_with_score(
        "far", k=1, filter={"first_letter": "f"}
    )
    output2 = await docsearch.asimilarity_search_with_score(
        "far", k=1, filter={"first_letter": "b"}
    )
    await docsearch.adelete_collection()
    assert output1 == [
        (Document(page_content="far", metadata={"first_letter": "f"}), 0.0)
    ]
    assert output2 == [
        (Document(page_content="bar", metadata={"first_letter": "b"}), 1.0)
    ]


def test_chroma_with_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    chroma_persist_dir = "./tests/persist_dir"
    collection_name = "test_collection"
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name=collection_name,
        texts=texts,
        embedding=FakeEmbeddings(),
        persist_directory=chroma_persist_dir,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory
    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        persist_directory=chroma_persist_dir,
    )
    output = docsearch.similarity_search("foo", k=1)

    # Clean up
    docsearch.delete_collection()

    # Persist doesn't need to be called again
    # Data will be automatically persisted on object deletion
    # Or on program exit


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_with_persistence_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search, with persistence."""
    chroma_persist_dir = "./tests/persist_dir"
    collection_name = "test_collection"
    texts = ["foo", "bar", "baz"]
    async_client = aclient
    docsearch = await Chroma.afrom_texts(
        client=async_client,
        collection_name=collection_name,
        texts=texts,
        embedding=FakeEmbeddings(),
        persist_directory=chroma_persist_dir,
    )

    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory
    docsearch = Chroma(
        client=async_client,
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        persist_directory=chroma_persist_dir,
    )
    await docsearch.aset_collection()
    output = await docsearch.asimilarity_search("foo", k=1)

    # Clean up
    await docsearch.adelete_collection()

    # Persist doesn't need to be called again
    # Data will be automatically persisted on object deletion
    # Or on program exit


def test_chroma_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    docsearch.delete_collection()
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_mmr_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = await docsearch.amax_marginal_relevance_search("foo", k=1)
    await docsearch.adelete_collection()
    assert output == [Document(page_content="foo")]


def test_chroma_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=embeddings
    )
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    docsearch.delete_collection()
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_mmr_by_vector_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
    )
    embedded_query = embeddings.embed_query("foo")
    output = await docsearch.amax_marginal_relevance_search_by_vector(
        embedded_query, k=1
    )
    await docsearch.adelete_collection()
    assert output == [Document(page_content="foo")]


def test_chroma_with_include_parameter() -> None:
    """Test end to end construction and include parameter."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    output1 = docsearch.get(include=["embeddings"])
    output2 = docsearch.get()
    docsearch.delete_collection()
    assert output1["embeddings"] is not None
    assert output2["embeddings"] is None


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_with_include_parameter_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Test end to end construction and include parameter."""
    texts = ["foo", "bar", "baz"]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output1 = await docsearch.aget(include=["embeddings"])
    output2 = await docsearch.aget()
    await docsearch.adelete_collection()
    assert output1["embeddings"] is not None
    assert output2["embeddings"] is None


def test_chroma_update_document() -> None:
    """Test the update_document function in the Chroma class."""
    # Make a consistent embedding
    embedding = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})

    # Initialize a Chroma instance with the original document
    docsearch = Chroma.from_documents(
        collection_name="test_collection",
        documents=[original_doc],
        embedding=embedding,
        ids=[document_id],
    )
    old_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(page_content=updated_content, metadata={"page": "0"})

    # Update the document in the Chroma instance
    docsearch.update_document(document_id=document_id, document=updated_doc)

    # Perform a similarity search with the updated content
    output = docsearch.similarity_search(updated_content, k=1)

    # Assert that the new embedding is correct
    new_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    docsearch.delete_collection()

    # Assert that the updated document is returned by the search
    assert output == [Document(page_content=updated_content, metadata={"page": "0"})]

    assert list(new_embedding) == list(embedding.embed_documents([updated_content])[0])
    assert list(new_embedding) != list(old_embedding)


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_update_document_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Test the update_document function in the Chroma class."""
    # Make a consistent embedding
    embedding = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})

    # Initialize a Chroma instance with the original document
    docsearch = await Chroma.afrom_documents(
        client=aclient,
        collection_name="test_collection",
        documents=[original_doc],
        embedding=embedding,
        ids=[document_id],
    )
    old_embedding = (await docsearch._collection.peek())["embeddings"][  # type: ignore
        (await docsearch._collection.peek())["ids"].index(document_id)  # type: ignore
    ]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(page_content=updated_content, metadata={"page": "0"})

    # Update the document in the Chroma instance
    await docsearch.aupdate_document(document_id=document_id, document=updated_doc)

    # Perform a similarity search with the updated content
    output = await docsearch.asimilarity_search(updated_content, k=1)

    # Assert that the new embedding is correct
    new_embedding = (await docsearch._collection.peek())["embeddings"][  # type: ignore
        (await docsearch._collection.peek())["ids"].index(document_id)  # type: ignore
    ]

    await docsearch.adelete_collection()

    # Assert that the updated document is returned by the search
    assert output == [Document(page_content=updated_content, metadata={"page": "0"})]

    assert list(new_embedding) == list(embedding.embed_documents([updated_content])[0])
    assert list(new_embedding) != list(old_embedding)


# TODO: RELEVANCE SCORE IS BROKEN. FIX TEST
def test_chroma_with_relevance_score_custom_normalization_fn() -> None:
    """Test searching with relevance score and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test1_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        relevance_score_fn=lambda d: d * 0,
        collection_metadata={"hnsw:space": "l2"},
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    docsearch.delete_collection()
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0),
    ]


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
# TODO: RELEVANCE SCORE IS BROKEN. FIX TEST
async def test_chroma_with_relevance_score_custom_normalization_fn_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Test searching with relevance score and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test1_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        relevance_score_fn=lambda d: d * 0,
        collection_metadata={"hnsw:space": "l2"},
    )
    output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
    await docsearch.adelete_collection()
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0),
    ]


def test_init_from_client(client: chromadb.ClientAPI) -> None:
    Chroma(client=client)


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_init_from_client_async(aclient: chromadb.AsyncClientAPI) -> None:
    Chroma(client=aclient)


def test_init_from_client_settings() -> None:
    import chromadb

    client_settings = chromadb.config.Settings()
    Chroma(client_settings=client_settings)


def test_chroma_add_documents_no_metadata() -> None:
    db = Chroma(embedding_function=FakeEmbeddings())
    db.add_documents([Document(page_content="foo")])

    db.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_add_documents_no_metadata_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    db = Chroma(client=aclient, embedding_function=FakeEmbeddings())
    await db.aset_collection()
    await db.aadd_documents([Document(page_content="foo")])
    await db.adelete_collection()


def test_chroma_add_documents_mixed_metadata() -> None:
    db = Chroma(embedding_function=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]
    actual_ids = db.add_documents(docs, ids=ids)
    search = db.similarity_search("foo bar")
    db.delete_collection()

    assert actual_ids == ids
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_chroma_add_documents_mixed_metadata_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    db = Chroma(client=aclient, embedding_function=FakeEmbeddings())
    await db.aset_collection()
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]
    actual_ids = await db.aadd_documents(docs, ids=ids)
    search = await db.asimilarity_search("foo bar")
    await db.adelete_collection()

    assert actual_ids == ids
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )


def batch_support_chroma_version() -> bool:
    major, minor, patch = chromadb.__version__.split(".")
    if int(major) == 0 and int(minor) >= 4 and int(patch) >= 10:
        return True
    return False


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
def test_chroma_large_batch() -> None:
    client = chromadb.HttpClient()
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore
    )
    docs = ["This is a test document"] * (client.get_max_batch_size() + 100)  # type: ignore
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

    db.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
async def test_chroma_large_batch_async(aclient: chromadb.AsyncClientAPI) -> None:
    async_client = aclient
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = await async_client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore
    )
    docs = ["This is a test document"] * (await async_client.get_max_batch_size() + 100)  # type: ignore
    db = await Chroma.afrom_texts(
        client=async_client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

    await db.adelete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
def test_chroma_large_batch_update() -> None:
    client = chromadb.HttpClient()
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore
    )
    docs = ["This is a test document"] * (client.get_max_batch_size() + 100)  # type: ignore
    ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=ids,
    )
    new_docs = [
        Document(
            page_content="This is a new test document", metadata={"doc_id": f"{i}"}
        )
        for i in range(len(docs) - 10)
    ]
    new_ids = [_id for _id in ids[: len(new_docs)]]
    db.update_documents(ids=new_ids, documents=new_docs)

    db.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
async def test_chroma_large_batch_update_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    async_client = aclient
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = await async_client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore
    )
    docs = ["This is a test document"] * (await async_client.get_max_batch_size() + 100)  # type: ignore
    ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db = await Chroma.afrom_texts(
        client=async_client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=ids,
    )
    new_docs = [
        Document(
            page_content="This is a new test document", metadata={"doc_id": f"{i}"}
        )
        for i in range(len(docs) - 10)
    ]
    new_ids = [_id for _id in ids[: len(new_docs)]]
    await db.aupdate_documents(ids=new_ids, documents=new_docs)

    await db.adelete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    batch_support_chroma_version(), reason="ChromaDB version does not support batching"
)
def test_chroma_legacy_batching() -> None:
    client = chromadb.HttpClient()
    embedding_function = Fak(size=255)
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=MyEmbeddingFunction,  # type: ignore
    )
    docs = ["This is a test document"] * 100
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

    db.delete_collection()


def test_create_collection_if_not_exist_default() -> None:
    """Tests existing behaviour without the new create_collection_if_not_exists flag."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    assert docsearch._client.get_collection("test_collection") is not None
    docsearch.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_create_collection_if_not_exist_default_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests existing behaviour without the new create_collection_if_not_exists flag."""
    texts = ["foo", "bar", "baz"]
    docsearch = await Chroma.afrom_texts(
        client=aclient,
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    assert await docsearch._client.get_collection("test_collection") is not None  # type: ignore
    await docsearch.adelete_collection()


def test_create_collection_if_not_exist_true_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection already existing."""
    client.create_collection("test_collection")
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )
    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_create_collection_if_not_exist_true_existing_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection already existing."""
    await aclient.create_collection("test_collection")
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )
    await vectorstore.aset_collection()
    assert await vectorstore._client.get_collection("test_collection") is not None  # type: ignore
    await vectorstore.adelete_collection()


def test_create_collection_if_not_exist_false_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection already existing."""
    client.create_collection("test_collection")
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=False,
    )
    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_create_collection_if_not_exist_false_existing_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection already existing."""
    await aclient.create_collection("test_collection")
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=False,
    )
    await vectorstore.aset_collection()
    assert await vectorstore._client.get_collection("test_collection") is not None  # type: ignore
    await vectorstore.adelete_collection()


def test_create_collection_if_not_exist_false_non_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection not-existing,
    should raise."""
    with pytest.raises(Exception, match="does not exist"):
        Chroma(
            client=client,
            collection_name="test_collection",
            embedding_function=FakeEmbeddings(),
            create_collection_if_not_exists=False,
        )


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_create_collection_if_not_exist_false_non_existing_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection not-existing,
    should raise."""
    with pytest.raises(Exception, match="does not exist"):
        vectorstore = Chroma(
            client=aclient,
            collection_name="test_collection",
            embedding_function=FakeEmbeddings(),
            create_collection_if_not_exists=False,
        )
        await vectorstore.aset_collection()


def test_create_collection_if_not_exist_true_non_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )

    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_create_collection_if_not_exist_true_non_existing_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )
    await vectorstore.aset_collection()
    assert await vectorstore._client.get_collection("test_collection") is not None  # type: ignore
    await vectorstore.adelete_collection()


def test_collection_none_after_delete(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )

    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()
    assert vectorstore._chroma_collection is None
    with pytest.raises(Exception, match="Chroma collection not initialized"):
        _ = vectorstore._collection
    with pytest.raises(Exception, match="does not exist"):
        vectorstore._client.get_collection("test_collection")
    with pytest.raises(Exception):
        vectorstore.similarity_search("foo")


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_collection_none_after_delete_async(
    aclient: chromadb.AsyncClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    await vectorstore.aset_collection()
    assert await vectorstore._client.get_collection("test_collection") is not None  # type: ignore
    await vectorstore.adelete_collection()
    assert vectorstore._chroma_collection is None
    with pytest.raises(Exception, match="Chroma collection not initialized"):
        _ = vectorstore._collection
    with pytest.raises(Exception, match="does not exist"):
        await vectorstore._client.get_collection("test_collection")  # type: ignore
    with pytest.raises(Exception):
        await vectorstore.asimilarity_search("foo")


def test_reset_collection(client: chromadb.ClientAPI) -> None:
    """Tests ensure_collection method."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    vectorstore.add_documents([Document(page_content="foo")])
    assert vectorstore._collection.count() == 1
    vectorstore.reset_collection()
    assert vectorstore._chroma_collection is not None
    assert vectorstore._client.get_collection("test_collection") is not None
    assert vectorstore._collection.name == "test_collection"
    assert vectorstore._collection.count() == 0
    # Clean up
    vectorstore.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_reset_collection_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Tests ensure_collection method."""
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    await vectorstore.aset_collection()
    await vectorstore.aadd_documents([Document(page_content="foo")])
    assert await vectorstore._collection.count() == 1  # type: ignore
    await vectorstore.areset_collection()
    assert vectorstore._chroma_collection is not None
    assert await vectorstore._client.get_collection("test_collection") is not None  # type: ignore
    assert vectorstore._collection.name == "test_collection"
    assert await vectorstore._collection.count() == 0  # type: ignore
    # Clean up
    await vectorstore.adelete_collection()


def test_delete_where_clause(client: chromadb.ClientAPI) -> None:
    """Tests delete_where_clause method."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    vectorstore.add_documents(
        [
            Document(page_content="foo", metadata={"test": "bar"}),
            Document(page_content="bar", metadata={"test": "foo"}),
        ]
    )
    assert vectorstore._collection.count() == 2
    vectorstore.delete(where={"test": "bar"})
    assert vectorstore._collection.count() == 1
    # Clean up
    vectorstore.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
async def test_delete_where_clause_async(aclient: chromadb.AsyncClientAPI) -> None:
    """Tests delete_where_clause method."""
    vectorstore = Chroma(
        client=aclient,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    await vectorstore.aset_collection()
    await vectorstore.aadd_documents(
        [
            Document(page_content="foo", metadata={"test": "bar"}),
            Document(page_content="bar", metadata={"test": "foo"}),
        ]
    )
    assert await vectorstore._collection.count() == 2  # type: ignore
    await vectorstore.adelete(where={"test": "bar"})
    assert await vectorstore._collection.count() == 1  # type: ignore
    # Clean up
    await vectorstore.adelete_collection()
