"""Test Marqo functionality."""
from typing import Dict

import marqo
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo

DEFAULT_MARQO_URL = "http://localhost:8882"
DEFAULT_MARQO_API_KEY = ""
INDEX_NAME = "langchain-integration-tests"


@pytest.fixture
def client() -> Marqo:
    # fixture for marqo client to be used throughout testing, resets the index
    client = marqo.Client(url=DEFAULT_MARQO_URL, api_key=DEFAULT_MARQO_API_KEY)
    try:
        client.index(INDEX_NAME).delete()
    except Exception:
        pass

    client.create_index(INDEX_NAME)
    return client


def test_marqo(client: Marqo) -> None:
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


def test_marqo_with_metadatas(client: Marqo) -> None:
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


def test_marqo_with_scores(client: Marqo) -> None:
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


def test_marqo_add_texts(client: Marqo) -> None:
    marqo_search = Marqo(client=client, index_name=INDEX_NAME)
    ids1 = marqo_search.add_texts(["1", "2", "3"])
    assert len(ids1) == 3
    ids2 = marqo_search.add_texts(["1", "2", "3"])
    assert len(ids2) == 3
    assert len(set(ids1).union(set(ids2))) == 6


def test_marqo_search(client: Marqo) -> None:
    marqo_search = Marqo(client=client, index_name=INDEX_NAME)
    input_documents = ["This is document 1", "2", "3"]
    ids = marqo_search.add_texts(input_documents)
    results = marqo_search.marqo_similarity_search("What is the first document?", k=3)
    assert len(ids) == len(input_documents)
    assert ids[0] == results["hits"][0]["_id"]


def test_marqo_bulk(client: Marqo) -> None:
    marqo_search = Marqo(client=client, index_name=INDEX_NAME)
    input_documents = ["This is document 1", "2", "3"]
    ids = marqo_search.add_texts(input_documents)
    bulk_results = marqo_search.bulk_similarity_search(
        ["What is the first document?", "2", "3"], k=3
    )

    assert len(ids) == len(input_documents)
    assert bulk_results[0][0].page_content == input_documents[0]
    assert bulk_results[1][0].page_content == input_documents[1]
    assert bulk_results[2][0].page_content == input_documents[2]


def test_marqo_weighted_query(client: Marqo) -> None:
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


def test_marqo_multimodal() -> None:
    client = marqo.Client(url=DEFAULT_MARQO_URL, api_key=DEFAULT_MARQO_API_KEY)
    try:
        client.index(INDEX_NAME).delete()
    except Exception:
        pass

    # reset the index for this example
    client.delete_index(INDEX_NAME)

    # This index could have been created by another system
    settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}
    client.create_index(INDEX_NAME, **settings)
    client.index(INDEX_NAME).add_documents(
        [
            # image of a bus
            {
                "caption": "Bus",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/"
                "examples/ImageSearchGuide/data/image4.jpg",
            },
            # image of a plane
            {
                "caption": "Plane",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/"
                "mainline/examples/ImageSearchGuide/data/image2.jpg",
            },
        ],
    )

    def get_content(res: Dict[str, str]) -> str:
        if "text" in res:
            return res["text"]
        return f"{res['caption']}: {res['image']}"

    marqo_search = Marqo(client, INDEX_NAME, page_content_builder=get_content)

    query = "vehicles that fly"
    docs = marqo_search.similarity_search(query)

    assert docs[0].page_content.split(":")[0] == "Plane"

    raised_value_error = False
    try:
        marqo_search.add_texts(["text"])
    except ValueError:
        raised_value_error = True

    assert raised_value_error
