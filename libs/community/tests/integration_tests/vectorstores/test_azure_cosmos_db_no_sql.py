"""Test AzureCosmosDBNoSqlVectorSearch functionality."""

import logging
import os
from time import sleep
from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
    Condition,
    CosmosDBQueryType,
    PreFilter,
)

logging.basicConfig(level=logging.DEBUG)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

# Host and Key for CosmosDB No SQl
HOST = os.environ.get("HOST")
KEY = os.environ.get("KEY")

database_name = "langchain_python_db"
container_name = "langchain_python_container"


@pytest.fixture()
def cosmos_client() -> Any:
    from azure.cosmos import CosmosClient

    return CosmosClient(HOST, KEY)


@pytest.fixture()
def partition_key() -> Any:
    from azure.cosmos import PartitionKey

    return PartitionKey(path="/id")


@pytest.fixture()
def azure_openai_embeddings() -> Any:
    openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        deployment=model_deployment, model=model_name, chunk_size=1
    )
    return openai_embeddings


def safe_delete_database(cosmos_client: Any) -> None:
    cosmos_client.delete_database(database_name)


def get_vector_indexing_policy(embedding_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": embedding_type}],
        "fullTextIndexes": [{"path": "/text"}],
    }


def get_vector_embedding_policy(
    distance_function: str, data_type: str, dimensions: int
) -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": data_type,
                "dimensions": dimensions,
                "distanceFunction": distance_function,
            }
        ]
    }


def get_full_text_policy() -> dict:
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": "/text", "language": "en-US"}],
    }


class TestAzureCosmosDBNoSqlVectorSearch:
    def test_from_documents_cosine_distance(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("intelligent herders", k=5)

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content
        safe_delete_database(cosmos_client)

    def test_from_texts_cosine_distance_delete_one(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: OpenAIEmbeddings,
    ) -> None:
        texts, metadatas = self._get_texts_and_metadata()

        store = AzureCosmosDBNoSqlVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("intelligent herders", k=1)
        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content

        # delete one document
        store.delete_document_by_id(str(output[0].metadata["id"]))
        sleep(2)

        output2 = store.similarity_search("intelligent herders", k=1)
        assert output2
        assert len(output2) == 1
        assert "Border Collies" not in output2[0].page_content
        safe_delete_database(cosmos_client)

    def test_from_documents_cosine_distance_with_filtering(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("intelligent herders", k=4)
        assert len(output) == 4
        assert "Border Collies" in output[0].page_content
        assert output[0].metadata["a"] == 1

        # pre_filter = {
        #     "conditions": [
        #         {"property": "metadata.a", "operator": "$eq", "value": 1},
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(property="metadata.a", operator="$eq", value=1),
            ],
        )
        output = store.similarity_search(
            "intelligent herders", k=4, pre_filter=pre_filter, with_embedding=True
        )

        assert len(output) == 3
        assert "Border Collies" in output[0].page_content
        assert output[0].metadata["a"] == 1

        # pre_filter = {
        #     "conditions": [
        #         {"property": "metadata.a", "operator": "$eq", "value": 1},
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(property="metadata.a", operator="$eq", value=1),
            ],
        )
        offset_limit = "OFFSET 0 LIMIT 1"

        output = store.similarity_search(
            "intelligent herders", k=4, pre_filter=pre_filter, offset_limit=offset_limit
        )

        assert len(output) == 1
        assert "Border Collies" in output[0].page_content
        assert output[0].metadata["a"] == 1
        safe_delete_database(cosmos_client)

    def test_from_documents_full_text_and_hybrid(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            full_text_policy=get_full_text_policy(),
            indexing_policy=get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            full_text_search_enabled=True,
        )

        sleep(480)  # waits for Cosmos DB to save contents to the collection

        # Full text search contains any
        # pre_filter = {
        #     "conditions": [
        #         {
        #             "property": "text",
        #             "operator": "$full_text_contains_any",
        #             "value": "intelligent herders",
        #         },
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(
                    property="text",
                    operator="$full_text_contains_all",
                    value="intelligent herders",
                ),
            ],
        )
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.FULL_TEXT_SEARCH,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search contains all
        # pre_filter = {
        #     "conditions": [
        #         {
        #             "property": "text",
        #             "operator": "$full_text_contains_all",
        #             "value": "intelligent herders",
        #         },
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(
                    property="text",
                    operator="$full_text_contains_all",
                    value="intelligent herders",
                ),
            ],
        )

        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.FULL_TEXT_SEARCH,
        )

        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking
        output = store.similarity_search(
            "intelligent herders", k=5, query_type=CosmosDBQueryType.FULL_TEXT_RANK
        )

        assert output
        assert len(output) == 5
        assert "Standard Poodles" in output[0].page_content

        # Full text search BM25 ranking with filtering
        # pre_filter = {
        #     "conditions": [
        #         {"property": "metadata.a", "operator": "$eq", "value": 1},
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(property="metadata.a", operator="$eq", value=1),
            ],
        )
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.FULL_TEXT_RANK,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking combination of full text search and vector search
        output = store.similarity_search(
            "intelligent herders", k=5, query_type=CosmosDBQueryType.HYBRID
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking with filtering
        # pre_filter = {
        #     "conditions": [
        #         {"property": "metadata.a", "operator": "$eq", "value": 1},
        #     ],
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(property="metadata.a", operator="$eq", value=1),
            ],
        )
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.HYBRID,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        # pre_filter = {
        #     "conditions": [
        #         {
        #             "property": "text",
        #             "operator": "$full_text_contains",
        #             "value": "energetic",
        #         },
        #     ]
        # }

        pre_filter = PreFilter(
            conditions=[
                Condition(
                    property="text", operator="$full_text_contains", value="energetic"
                ),
            ],
        )
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.FULL_TEXT_RANK,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        # pre_filter = {
        #     "conditions": [
        #         {
        #             "property": "text",
        #             "operator": "$full_text_contains",
        #             "value": "energetic",
        #         },
        #         {"property": "metadata.a", "operator": "$eq", "value": 2},
        #     ],
        #     "logical_operator": "$and",
        # }
        pre_filter = PreFilter(
            conditions=[
                Condition(
                    property="text", operator="$full_text_contains", value="energetic"
                ),
                Condition(property="metadata.a", operator="$eq", value=2),
            ],
            logical_operator="$and",
        )
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            pre_filter=pre_filter,
            query_type=CosmosDBQueryType.FULL_TEXT_RANK,
        )

        assert output
        assert len(output) == 2
        assert "Standard Poodles" in output[0].page_content

    def _get_documents(self) -> List[Document]:
        return [
            Document(
                page_content="Border Collies are intelligent, energetic "
                "herders skilled in outdoor activities.",
                metadata={"a": 1},
            ),
            Document(
                page_content="Golden Retrievers are friendly, loyal companions "
                "with excellent retrieving skills.",
                metadata={"a": 2},
            ),
            Document(
                page_content="Labrador Retrievers are playful, eager "
                "learners and skilled retrievers.",
                metadata={"a": 1},
            ),
            Document(
                page_content="Australian Shepherds are agile, energetic "
                "herders excelling in outdoor tasks.",
                metadata={"a": 2, "b": 1},
            ),
            Document(
                page_content="German Shepherds are brave, loyal protectors "
                "excelling in versatile tasks.",
                metadata={"a": 1, "b": 2},
            ),
            Document(
                page_content="Standard Poodles are intelligent, energetic "
                "learners excelling in agility.",
                metadata={"a": 2, "b": 3},
            ),
        ]

    def _get_texts_and_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts = [
            "Border Collies are intelligent, "
            "energetic herders skilled in outdoor activities.",
            "Golden Retrievers are friendly, "
            "loyal companions with excellent retrieving skills.",
            "Labrador Retrievers are playful, "
            "eager learners and skilled retrievers.",
            "Australian Shepherds are agile, "
            "energetic herders excelling in outdoor tasks.",
            "German Shepherds are brave, "
            "loyal protectors excelling in versatile tasks.",
            "Standard Poodles are intelligent, "
            "energetic learners excelling in agility.",
        ]
        metadatas = [
            {"a": 1},
            {"a": 2},
            {"a": 1},
            {"a": 2, "b": 1},
            {"a": 1, "b": 2},
            {"a": 2, "b": 1},
        ]
        return texts, metadatas
