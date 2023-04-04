"""Test Marqo functionality."""
import marqo
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo

DEFAULT_MARQO_URL = "http://localhost:8882"
DEFAULT_MARQO_API_KEY = ""
INDEX_NAME = "langchain-integration-tests"


@pytest.fixture
def client():
    # fixture for marqo client to be used throughout testing, resets the index
    client = marqo.Client(url=DEFAULT_MARQO_URL, api_key=DEFAULT_MARQO_API_KEY)
    try:
        client.index(INDEX_NAME).delete()
    except Exception:
        pass

    client.create_index(INDEX_NAME)
    return client


def test_marqo(client) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    marqo_search = Marqo.from_texts(
        texts=texts,
        index_name=INDEX_NAME,
        url=DEFAULT_MARQO_URL,
        api_key=DEFAULT_MARQO_API_KEY,
        verbose=False,
    )
    results = marqo_search.similarity_search("foo", k=1)
    assert results == [Document(page_content="foo")]


def test_marqo_with_metadatas(client) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    marqo_search = Marqo.from_texts(
        texts=texts,
        metadatas=metadatas,
        index_name=INDEX_NAME,
        url=DEFAULT_MARQO_URL,
        api_key=DEFAULT_MARQO_API_KEY,
        verbose=False,
    )
    results = marqo_search.similarity_search("foo", k=1)
    assert results == [Document(page_content="foo", metadata={"page": 0})]


def test_marqo_with_scores(client) -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    marqo_search = Marqo.from_texts(
        texts=texts,
        metadatas=metadatas,
        index_name=INDEX_NAME,
        url=DEFAULT_MARQO_URL,
        api_key=DEFAULT_MARQO_API_KEY,
        verbose=False,
    )
    results = marqo_search.similarity_search_with_score("foo", k=3)
    docs = [r[0] for r in results]
    scores = [r[1] for r in results]

    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] > scores[1] > scores[2]


def test_marqo_add_texts(client) -> None:
    marqo_search = Marqo(client=client, index_name=INDEX_NAME)
    ids1 = marqo_search.add_texts(["1", "2", "3"])
    assert len(ids1) == 3
    ids2 = marqo_search.add_texts(["1", "2", "3"])
    assert len(ids2) == 3
    assert len(set(ids1).union(set(ids2))) == 6


def test_marqo_search(client) -> None:
    marqo_search = Marqo(client=client, index_name=INDEX_NAME)
    input_documents = ["This is document 1", "2", "3"]
    ids = marqo_search.add_texts(input_documents)
    hits = marqo_search.marqo_similarity_search("What is the first document?", k=3)
    assert len(ids) == len(input_documents)
    assert ids[0] == hits[0]["_id"]


def test_marqo_weighted_query(client) -> None:
    """Test end to end construction and search."""
    texts = ["Smartphone", "Telephone"]
    marqo_search = Marqo.from_texts(
        texts=texts,
        index_name=INDEX_NAME,
        url=DEFAULT_MARQO_URL,
        api_key=DEFAULT_MARQO_API_KEY,
        verbose=False,
    )
    results = marqo_search.similarity_search(
        {"communications device": 1.0, "Old technology": -5.0}, k=1
    )
    assert results == [Document(page_content="Smartphone")]
