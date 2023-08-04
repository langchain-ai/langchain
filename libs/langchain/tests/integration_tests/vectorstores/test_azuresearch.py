import os
import time

import openai
import pytest
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

load_dotenv()

# Azure OpenAI settings
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE", "")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY", "")
model: str = os.getenv("OPENAI_EMBEDDINGS_ENGINE_DOC", "text-embedding-ada-002")

# Vector store settings
vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
index_name: str = "embeddings-vector-store-test"


@pytest.fixture
def similarity_search_test() -> None:
    """Test end to end construction and search."""
    # Create Embeddings
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
    # Create Vector store
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    # Add texts to vector store and perform a similarity search
    vector_store.add_texts(
        ["Test 1", "Test 2", "Test 3"],
        [
            {"title": "Title 1", "any_metadata": "Metadata 1"},
            {"title": "Title 2", "any_metadata": "Metadata 2"},
            {"title": "Title 3", "any_metadata": "Metadata 3"},
        ],
    )
    time.sleep(1)
    res = vector_store.similarity_search(query="Test 1", k=3)
    assert len(res) == 3


def from_text_similarity_search_test() -> None:
    """Test end to end construction and search."""
    # Create Embeddings
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
    # Create Vector store
    vector_store: AzureSearch = AzureSearch.from_texts(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        texts=["Test 1", "Test 2", "Test 3"],
        embedding=embeddings,
    )
    time.sleep(1)
    # Perform a similarity search
    res = vector_store.similarity_search(query="Test 1", k=3)
    assert len(res) == 3


def test_semantic_hybrid_search() -> None:
    """Test end to end construction and search."""
    # Create Embeddings
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
    # Create Vector store
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default",
    )
    # Add texts to vector store and perform a semantic hybrid search
    vector_store.add_texts(
        ["Test 1", "Test 2", "Test 3"],
        [
            {"title": "Title 1", "any_metadata": "Metadata 1"},
            {"title": "Title 2", "any_metadata": "Metadata 2"},
            {"title": "Title 3", "any_metadata": "Metadata 3"},
        ],
    )
    time.sleep(1)
    res = vector_store.semantic_hybrid_search(query="What's Azure Search?", k=3)
    assert len(res) == 3
