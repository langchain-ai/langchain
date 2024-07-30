import os
from time import monotonic, sleep
from typing import Any, Dict, List, Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from langchain_mongodb import MongoDBAtlasVectorSearch, index
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "test_retrievers"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
PAGE_CONTENT_FIELD = "text"
SEARCH_INDEX_NAME = "text_index"

DIMENSIONS = 1536
TIMEOUT = 120.0
INTERVAL = 0.5


@pytest.fixture
def example_documents():
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
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small",
        )
    except Exception:
        return ConsistentFakeEmbeddings(DIMENSIONS)


@pytest.fixture
def collection_with_two_indexes() -> Collection:
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
        index.create_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
            wait_until_complete=TIMEOUT,
        )

    return clxn


def test_retriever(
    embedding_openai: Embeddings,
    collection_with_two_indexes: Collection,
    example_documents,
) -> None:
    """Demonstrate usage and parity of VectorStore similarity_search with Retriever.invoke."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection_with_two_indexes,
        embedding=embedding_openai,
        vector_index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    vectorstore.add_documents(example_documents)

    sleep(TIMEOUT)  # Wait for documents to be sync'd

    retriever = MongoDBAtlasHybridSearchRetriever(
        collection=collection_with_two_indexes,
        embedding_model=embedding_openai,
        vector_search_index_name=VECTOR_INDEX_NAME,
        search_index_name=SEARCH_INDEX_NAME,
        page_content_field=PAGE_CONTENT_FIELD,
        embedding_field=EMBEDDING_FIELD,
        top_k=3,
    )

    query1 = "What was the latest city that I visited?"
    results = retriever.invoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content
