"""Test SQLServer_VectorStore functionality."""

import os
from typing import List

import pytest
from langchain_core.documents import Document
from sqlalchemy.exc import DBAPIError

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import SQLServer_VectorStore

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING"))


@pytest.fixture
def texts() -> List[str]:
    """Definition of texts used in the tests."""
    query = [
        """I have bought several of the Vitality canned dog food products and have 
        found them all to be of good quality. The product looks more like a stew 
        than a processed meat and it smells better. My Labrador is finicky and she 
        appreciates this product better than  most.""",
        """The candy is just red , No flavor . Just  plan and chewy .
        I would never buy them again""",
        "Arrived in 6 days and were so stale i could not eat any of the 6 bags!!",
        """Got these on sale for roughly 25 cents per cup, which is half the price 
        of my local grocery stores, plus they rarely stock the spicy flavors. These 
        things are a GREAT snack for my office where time is constantly crunched and 
        sometimes you can't escape for a real meal. This is one of my favorite flavors 
        of Instant Lunch and will be back to buy every time it goes on sale.""",
        """If you are looking for a less messy version of licorice for the children, 
        then be sure to try these!  They're soft, easy to chew, and they don't get your 
        hands all sticky and gross in the car, in the summer, at the beach, etc. 
        We love all the flavos and sometimes mix these in with the chocolate to have a 
        very nice snack! Great item, great price too, highly recommend!""",
    ]
    return query  # provide this data to the test.


@pytest.fixture
def metadatas() -> List[dict]:
    """Definition of metadatas used in the tests."""
    query_metadata = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 3, "summary": "stale product"},
        {"id": 4, "summary": "Great value and convenient ramen"},
        {"id": 5, "summary": "Great for the kids!"},
    ]
    return query_metadata  # provide this data to the test.


@pytest.fixture
def store() -> SQLServer_VectorStore:
    """Setup resources that are needed for the duration of the test."""
    store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests",
    )
    return store  # provide this data to the test


@pytest.fixture
def docs() -> List[Document]:
    """Definition of doc variable used in the tests."""
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
            metadata={"color": "red", "type": "fruit", "length": 6},
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(
            page_content="cat", metadata={"color": "black", "type": "pet", "length": 3}
        ),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    return docs  # provide this data to the test


def test_sqlserver_add_texts(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that add text returns equivalent number of ids of input texts."""
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_no_metadata_is_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that when user calls the add_texts function without providing metadata,
    the embedded text still get added to the vector store."""
    result = store.add_texts(texts)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_text_length_and_metadata_length_vary(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that all texts provided are added into the vector store
    even when metadata is not available for all the texts."""
    # We get all metadatas except the last one from our metadatas fixture.
    # The text without a corresponding metadata should be added to the vector store.
    metadatas = metadatas[:-1]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_list_of_given_id_is_less_than_list_of_texts(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that when length of given id is less than length of texts,
    random ids are created."""
    # List of ids is one less than len(texts) which is 5.
    metadatas = metadatas[:-1]
    metadatas.append({"summary": "Great for the kids!"})
    result = store.add_texts(texts, metadatas)
    # Length of ids returned by add_texts function should be equal to length of texts.
    assert len(result) == len(texts)


def test_add_document_with_sqlserver(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that when add_document function is used, it integrates well
    with the add_text function in SQLServer Vector Store."""
    result = store.add_documents(docs)
    assert len(result) == len(docs)


def test_that_a_document_entry_without_metadata_will_be_added_to_vectorstore(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that you can add a document that has no metadata into the vectorstore."""
    documents = docs[:-1]
    documents.append(Document(page_content="elderberry"))
    result = store.add_documents(documents)
    assert len(result) == len(documents)


def test_that_similarity_search_returns_expected_no_of_documents(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that the amount of documents returned when similarity search
    is called is the same as the number of documents requested."""
    store.add_texts(texts)
    result = store.similarity_search(query="Good review", k=3)
    assert len(result) == 3


def test_that_similarity_search_returns_results_with_scores_sorted_in_ascending_order(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Assert that the list returned by a similarity search
    is sorted in an ascending order. The implication is that
    we have the smallest score (most similar doc.) returned first.
    """
    store.add_texts(texts)
    result_with_score = store.similarity_search_with_score("Good review", k=4)
    assert result_with_score == sorted(result_with_score, key=lambda x: x[1])
