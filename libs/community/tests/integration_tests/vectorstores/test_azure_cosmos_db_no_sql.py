"""Test AzureCosmosDBVectorSearch functionality."""
import logging
import os
from typing import Any
from time import sleep

import pytest
from azure.cosmos import CosmosClient, DatabaseProxy

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch

logging.basicConfig(level=logging.DEBUG)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

# Host and Key for CosmosDB No SQl
HOST = os.environ.get("HOST")
KEY = os.environ.get("KEY")

cosmos_client = CosmosClient(HOST, KEY)
database_name = "langchain_python_db"
container_name = "langchain_python_container"
partition_key = "/description"
cosmos_container_properties = {"partition_key": partition_key}


@pytest.fixture()
def azure_openai_embeddings() -> Any:
    openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        deployment=model_deployment, model=model_name, chunk_size=1
    )
    return openai_embeddings


def safe_delete_database() -> None:
    cosmos_client.delete_database(database_name)


def get_vector_indexing_policy(embedding_type):
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": f"{embedding_type}"}],
    }


def get_vector_embedding_policy(distance_function, data_type):
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": f"{data_type}",
                "dimensions": 1536,
                "distanceFunction": f"{distance_function}",
            }
        ]
    }


class TestAzureCosmosDBNoSqlVectorSearch:

    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=cosmos_client,
            database_name=database_name,
            partition_key=partition_key,
            vector_embedding_policy=get_vector_embedding_policy("", ""),
            indexing_policy=get_vector_indexing_policy(""),
            cosmos_container_properties=cosmos_container_properties,
        )

    def test_from_documents_cosine_distance(
            self, azure_openai_embeddings: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            partition_key=partition_key,
            vector_embedding_policy=get_vector_embedding_policy("", ""),
            indexing_policy=get_vector_indexing_policy(""),
            cosmos_container_properties=cosmos_container_properties,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search(
            "Sandwich",
            k=1
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        safe_delete_database()

    def test_from_texts_cosine_distance_delete_one(self, azure_openai_embeddings: OpenAIEmbeddings) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]

        store = AzureCosmosDBNoSqlVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            partition_key=partition_key,
            vector_embedding_policy=get_vector_embedding_policy("", ""),
            indexing_policy=get_vector_indexing_policy(""),
            cosmos_container_properties=cosmos_container_properties,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search(
            "Sandwich",
            k=1
        )
        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        # delete one document
        store.delete_document_by_id(str(output[0]["id"]))
        sleep(2)

        output2 = store.similarity_search(
            "Sandwich",
            k=1
        )
        assert output2
        assert output2[0].page_content != "What is a sandwich?"
        safe_delete_database()