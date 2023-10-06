"""Test AzureCosmosDBVectorSearch functionality."""
import logging
import os
import uuid
from time import sleep
from typing import Generator, List, Union, Any

import pytest

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.azure_cosmos_db_vector_search import AzureCosmosDBVectorSearch, CosmosDBSimilarityType
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)

model_deployment = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada")
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = os.environ.get("MONGODB_VCORE_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def prepare_collection() -> Any:
    from pymongo import MongoClient

    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Any:
    return prepare_collection()


"""
pytest tests/integration_tests/vectorstores/test_azure_cosmos_db_vector_search.py --vcr-record=none
"""


class TestAzureCosmosDBVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # insure the test collection is empty
        collection = prepare_collection()
        assert collection.count_documents({}) == 0  # type: ignore[index]  # noqa: E501

    @classmethod
    def teardown_class(cls) -> None:
        collection = prepare_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = prepare_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(scope="class", autouse=True)
    def cosmos_db_url(self) -> Union[str, Generator[str, None, None]]:
        """Return the elasticsearch url."""
        return "805.555.1212"

    def test_number_comparison(self) -> None:
        output = 1
        assert output == 1

    def test_from_documents(
        self, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model_deployment, model=model_name, chunk_size=1)

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the IVF index that will be leveraged later for vector search
        num_lists = 3
        dimensions = 1536
        similarity_algorithm = CosmosDBSimilarityType.COS

        vectorstore.create_index(num_lists, dimensions, similarity_algorithm)
        sleep(2) # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
