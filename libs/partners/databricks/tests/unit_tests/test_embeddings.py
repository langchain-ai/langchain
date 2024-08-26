"""Test Together AI embeddings."""

from typing import Any, Dict, Generator
from unittest import mock

import pytest
from mlflow.deployments import BaseDeploymentClient  # type: ignore[import-untyped]

from langchain_databricks import DatabricksEmbeddings


def _mock_embeddings(endpoint: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": list(range(1536)),
                "index": 0,
            }
            for _ in inputs["input"]
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


@pytest.fixture
def mock_client() -> Generator:
    client = mock.MagicMock()
    client.predict.side_effect = _mock_embeddings
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=client):
        yield client


@pytest.fixture
def embeddings() -> DatabricksEmbeddings:
    return DatabricksEmbeddings(
        endpoint="text-embedding-3-small",
        documents_params={"fruit": "apple"},
        query_params={"fruit": "banana"},
    )


def test_embed_documents(
    mock_client: BaseDeploymentClient, embeddings: DatabricksEmbeddings
) -> None:
    documents = ["foo"] * 30
    output = embeddings.embed_documents(documents)
    assert len(output) == 30
    assert len(output[0]) == 1536
    assert mock_client.predict.call_count == 2
    assert all(
        call_arg[1]["inputs"]["fruit"] == "apple"
        for call_arg in mock_client().predict.call_args_list
    )


def test_embed_query(
    mock_client: BaseDeploymentClient, embeddings: DatabricksEmbeddings
) -> None:
    query = "foo bar"
    output = embeddings.embed_query(query)
    assert len(output) == 1536
    mock_client.predict.assert_called_once()
    assert mock_client.predict.call_args[1] == {
        "endpoint": "text-embedding-3-small",
        "inputs": {"input": [query], "fruit": "banana"},
    }
