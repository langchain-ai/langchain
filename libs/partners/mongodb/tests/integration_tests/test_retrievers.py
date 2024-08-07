import os
from time import sleep
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import index
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
)

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "test_retrievers"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
PAGE_CONTENT_FIELD = "text"
SEARCH_INDEX_NAME = "text_index"

DIMENSIONS = 1536
TIMEOUT = 60.0
INTERVAL = 0.5


@pytest.fixture
def example_documents() -> List[Document]:
    return [
        Document(page_content="In 2023, I visited Paris"),
        Document(page_content="In 2022, I visited New York"),
        Document(page_content="In 2021, I visited New Orleans"),
        Document(page_content="Sandwiches are beautiful. Sandwiches are fine."),
    ]


@pytest.fixture
def embedding_openai() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    try:
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
            model="text-embedding-3-small",
        )
    except Exception:
        return ConsistentFakeEmbeddings(DIMENSIONS)


@pytest.fixture
def collection() -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    client: MongoClient = MongoClient(CONNECTION_STRING)
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if not any([VECTOR_INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        index.create_vector_search_index(
            collection=clxn,
            index_name=VECTOR_INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
            wait_until_complete=TIMEOUT,
        )

    if not any([SEARCH_INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        index.create_fulltext_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
            wait_until_complete=TIMEOUT,
        )

    return clxn


def test_hybrid_retriever(
    embedding_openai: Embeddings,
    collection: Collection,
    example_documents: List[Document],
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_openai,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    vectorstore.add_documents(example_documents)

    sleep(TIMEOUT)  # Wait for documents to be sync'd

    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        top_k=3,
    )

    query1 = "What was the latest city that I visited?"
    results = retriever.invoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_fulltext_retriever(
    collection: Collection,
    example_documents: List[Document],
) -> None:
    """Test result of performing fulltext search

    Independent of the VectorStore, one adds documents
    via MongoDB's Collection API
    """
    #

    collection.insert_many(
        [{PAGE_CONTENT_FIELD: doc.page_content} for doc in example_documents]
    )
    sleep(TIMEOUT)  # Wait for documents to be sync'd

    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name=SEARCH_INDEX_NAME,
        search_field=PAGE_CONTENT_FIELD,
    )

    query = "When was the last time I visited new orleans?"
    results = retriever.invoke(query)
    assert "New Orleans" in results[0].page_content
    assert "score" in results[0].metadata


def test_vector_retriever(
    embedding_openai: Embeddings,
    collection: Collection,
    example_documents: List[Document],
) -> None:
    """Test VectorStoreRetriever"""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_openai,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    vectorstore.add_documents(example_documents)

    sleep(TIMEOUT)  # Wait for documents to be sync'd

    retriever = vectorstore.as_retriever()

    query1 = "What was the latest city that I visited?"
    results = retriever.invoke(query1)
    assert len(results) == 4
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content
