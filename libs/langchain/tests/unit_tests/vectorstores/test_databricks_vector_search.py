from typing import List, Optional
from unittest.mock import patch, MagicMock

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import DatabricksVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)

TEST_DIMENSION = 1576


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (TEST_DIMENSION - 1) + [float(i)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (TEST_DIMENSION - 1) + [float(0.0)]


@pytest.fixture(scope="function", autouse=True)
def delta_sync_with_managed_embeddings_index():
    from databricks.vector_search.client import VectorSearchIndex

    index = MagicMock(spec=VectorSearchIndex)
    index.describe.return_value = {
        "name": "ml.llm.index",
        "endpoint_name": "vector_search_endpoint",
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "ml.llm.source_table",
            "pipeline_type": "CONTINUOUS",
            "embedding_source_columns": [
                {
                    "name": "text",
                    "embedding_model_endpoint_name": "openai-text-embedding",
                }
            ],
        },
    }

    yield index


@pytest.fixture(scope="function", autouse=True)
def delta_sync_with_self_managed_embeddings_index():
    from databricks.vector_search.client import VectorSearchIndex

    index = MagicMock(spec=VectorSearchIndex)
    index.describe.return_value = {
        "name": "ml.llm.index",
        "endpoint_name": "vector_search_endpoint",
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "ml.llm.source_table",
            "pipeline_type": "CONTINUOUS",
            "embedding_vector_columns": [
                {
                    "name": "text_vector",
                    "embedding_dimension": TEST_DIMENSION,
                }
            ],
        },
    }

    yield index


@pytest.fixture(scope="function", autouse=True)
def direct_access_index():
    from databricks.vector_search.client import VectorSearchIndex

    index = MagicMock(spec=VectorSearchIndex)
    index.describe.return_value = {
        "name": "ml.llm.index",
        "endpoint_name": "vector_search_endpoint",
        "index_type": "DIRECT_ACCESS",
        "primary_key": "id",
        "direct_access_index_spec": {
            "embedding_vector_columns": [
                {
                    "name": "text_vector",
                    "embedding_dimension": TEST_DIMENSION,
                }
            ],
            "schema_json": '{"id": "int", "feat1": "str", "feat2": "float", "text": "string", "text_vector": "array<float>"}',
        },
    }

    yield index


@pytest.mark.requires("databricks.vector_search")
def test_init_delta_sync_with_managed_embeddings_index(
    delta_sync_with_managed_embeddings_index
):
    vectorsearch = DatabricksVectorSearch(delta_sync_with_managed_embeddings_index)
    assert vectorsearch.index == delta_sync_with_managed_embeddings_index


@pytest.mark.requires("databricks.vector_search")
def test_init_delta_sync_with_self_managed_embeddings_index(
    delta_sync_with_self_managed_embeddings_index
):
    vectorsearch = DatabricksVectorSearch(
        delta_sync_with_self_managed_embeddings_index,
        embedding=FakeEmbeddingsWithDimension(),
        text_column="text",
    )
    assert vectorsearch.index == delta_sync_with_self_managed_embeddings_index


@pytest.mark.requires("databricks.vector_search")
def test_init_direct_access_index(direct_access_index):
    vectorsearch = DatabricksVectorSearch(
        direct_access_index,
        embedding=FakeEmbeddingsWithDimension(),
        text_column="text",
    )
    assert vectorsearch.index == direct_access_index
