"""Test AzureCosmosDBVectorSearch functionality."""

import logging
import os
from time import sleep
from typing import Any, Generator, Optional, Union

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)

logging.basicConfig(level=logging.DEBUG)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

INDEX_NAME = "langchain-test-index"
INDEX_NAME_VECTOR_HNSW = "langchain-test-index-hnsw"
INDEX_NAME_VECTOR_DISKANN = "langchain-test-index-diskann"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING: str = os.environ.get("MONGODB_VCORE_URI", "")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

num_lists = 3
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1
maxDegree = 50
lBuild = 40
lSearch = 100
application_name = "LANGCHAIN_PYTHON"


def prepare_collection() -> Any:
    from pymongo import MongoClient

    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Any:
    return prepare_collection()


@pytest.fixture()
def azure_openai_embeddings() -> Any:
    openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        model=model_name,
        chunk_size=1,
    )
    return openai_embeddings


"""
This is how to run the integration tests:

cd libs/langchain
pytest tests/integration_tests/vectorstores/test_azure_cosmos_db.py 
"""


class TestAzureCosmosDBVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set")
        if not os.getenv("AZURE_OPENAI_API_VERSION"):
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable is not set")

        # insure the test collection is empty
        collection = prepare_collection()
        assert collection.count_documents({}) == 0  # type: ignore[index]

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
        """Return the cosmos db url."""
        return "805.555.1212"

    def test_from_documents_cosine_distance(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME,
            application_name=application_name,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_documents_inner_product(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, CosmosDBSimilarityType.IP, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_cosine_distance(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, CosmosDBSimilarityType.IP, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output[0].page_content == "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_cosine_distance(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_one(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        first_document_id_object = output[0].metadata["_id"]
        first_document_id = str(first_document_id_object)

        vectorstore.delete_document_by_id(first_document_id)
        sleep(2)  # waits for the index to be updated

        output2 = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )
        assert output2
        assert output2[0].page_content != "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_multiple(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        first_document_id = str(output[0].metadata["_id"])

        second_document_id = str(output[1].metadata["_id"])

        third_document_id = str(output[2].metadata["_id"])

        document_ids = [first_document_id, second_document_id, third_document_id]
        vectorstore.delete(document_ids)
        sleep(2)  # waits for the index to be updated

        output_2 = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )
        assert output
        assert output_2

        assert len(output) == 4  # we should see all the four documents
        assert (
            len(output_2) == 1
        )  # we should see only one document left after three have been deleted

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_inner_product(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, CosmosDBSimilarityType.IP, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_euclidean_distance(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, CosmosDBSimilarityType.L2, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=kind,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_max_marginal_relevance_cosine_distance(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=kind,
            lambda_mult=0.1,
            score_threshold=score_threshold,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    def test_max_marginal_relevance_inner_product(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists, dimensions, CosmosDBSimilarityType.IP, kind, m, ef_construction
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=kind,
            lambda_mult=0.1,
            score_threshold=score_threshold,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    """
        Test cases for the similarity algorithm using vector-hnsw
    """

    def test_from_documents_cosine_distance_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_documents_inner_product_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_cosine_distance_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output[0].page_content == "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_cosine_distance_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_one_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        first_document_id_object = output[0].metadata["_id"]
        first_document_id = str(first_document_id_object)

        vectorstore.delete_document_by_id(first_document_id)
        sleep(2)  # waits for the index to be updated

        output2 = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )
        assert output2
        assert output2[0].page_content != "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_multiple_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        first_document_id = str(output[0].metadata["_id"])

        second_document_id = str(output[1].metadata["_id"])

        third_document_id = str(output[2].metadata["_id"])

        document_ids = [first_document_id, second_document_id, third_document_id]
        vectorstore.delete(document_ids)
        sleep(2)  # waits for the index to be updated

        output_2 = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )
        assert output
        assert output_2

        assert len(output) == 4  # we should see all the four documents
        assert (
            len(output_2) == 1
        )  # we should see only one document left after three have been deleted

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_inner_product_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            ef_search=ef_search,
            score_threshold=score_threshold,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_max_marginal_relevance_cosine_distance_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            lambda_mult=0.1,
            score_threshold=score_threshold,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    def test_max_marginal_relevance_inner_product_vector_hnsw(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_HNSW,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            num_lists,
            dimensions,
            similarity_algorithm,
            CosmosDBVectorSearchType.VECTOR_HNSW,
            m,
            ef_construction,
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            lambda_mult=0.1,
            score_threshold=score_threshold,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    """
        Test cases for the similarity algorithm using vector-diskann
    """

    def test_from_documents_cosine_distance_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_documents_inner_product_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = AzureCosmosDBVectorSearch.from_documents(
            documents,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_cosine_distance_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output[0].page_content == "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_cosine_distance_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_one_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        first_document_id_object = output[0].metadata["_id"]
        first_document_id = str(first_document_id_object)

        vectorstore.delete_document_by_id(first_document_id)
        sleep(2)  # waits for the index to be updated

        output2 = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )
        assert output2
        assert output2[0].page_content != "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_multiple_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        first_document_id = str(output[0].metadata["_id"])

        second_document_id = str(output[1].metadata["_id"])

        third_document_id = str(output[2].metadata["_id"])

        document_ids = [first_document_id, second_document_id, third_document_id]
        vectorstore.delete(document_ids)
        sleep(2)  # waits for the index to be updated

        output_2 = vectorstore.similarity_search(
            "Sandwich",
            k=5,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )
        assert output
        assert output_2

        assert len(output) == 4  # we should see all the four documents
        assert (
            len(output_2) == 1
        )  # we should see only one document left after three have been deleted

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_inner_product_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search(
            "Sandwich",
            k=1,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lSearch=lSearch,
        )

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_max_marginal_relevance_cosine_distance_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the IVF index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lambda_mult=0.1,
            lSearch=lSearch,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    def test_max_marginal_relevance_inner_product_vector_diskann(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = AzureCosmosDBVectorSearch.from_texts(
            texts,
            azure_openai_embeddings,
            collection=collection,
            index_name=INDEX_NAME_VECTOR_DISKANN,
        )

        # Create the DiskANN index that will be leveraged later for vector search
        vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            max_degree=maxDegree,
            l_build=lBuild,
        )
        sleep(2)  # waits for the index to be set up

        query = "foo"
        output = vectorstore.max_marginal_relevance_search(
            query,
            k=10,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            lambda_mult=0.1,
            lSearch=lSearch,
            with_embedding=True,
        )

        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

        vectorstore.delete_index()

    @staticmethod
    def invoke_delete_with_no_args(
        azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> Optional[bool]:
        vectorstore: AzureCosmosDBVectorSearch = (
            AzureCosmosDBVectorSearch.from_connection_string(
                CONNECTION_STRING,
                NAMESPACE,
                azure_openai_embeddings,
                index_name=INDEX_NAME,
                application_name=application_name,
            )
        )

        return vectorstore.delete()

    @staticmethod
    def invoke_delete_by_id_with_no_args(
        azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        vectorstore: AzureCosmosDBVectorSearch = (
            AzureCosmosDBVectorSearch.from_connection_string(
                CONNECTION_STRING,
                NAMESPACE,
                azure_openai_embeddings,
                index_name=INDEX_NAME,
                application_name=application_name,
            )
        )

        vectorstore.delete_document_by_id()

    def test_invalid_arguments_to_delete(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(ValueError) as exception_info:
            self.invoke_delete_with_no_args(azure_openai_embeddings, collection)
        assert str(exception_info.value) == "No document ids provided to delete."

    def test_no_arguments_to_delete_by_id(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(Exception) as exception_info:
            self.invoke_delete_by_id_with_no_args(
                azure_openai_embeddings=azure_openai_embeddings, collection=collection
            )
        assert str(exception_info.value) == "No document id provided to delete."
