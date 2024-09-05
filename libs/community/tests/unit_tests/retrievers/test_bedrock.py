from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_retriever_config() -> dict:
    return {"vectorSearchConfiguration": {"numberOfResults": 4}}


@pytest.fixture
def amazon_retriever(
    mock_client: MagicMock, mock_retriever_config: dict
) -> AmazonKnowledgeBasesRetriever:
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test_kb_id",
        retrieval_config=mock_retriever_config,  # type: ignore[arg-type]
        client=mock_client,
    )


def test_create_client() -> None:
    # Import error if boto3 is not installed
    # Value error if credentials are not supplied.
    with pytest.raises((ImportError, ValueError)):
        AmazonKnowledgeBasesRetriever()  # type: ignore


def test_standard_params(amazon_retriever: AmazonKnowledgeBasesRetriever) -> None:
    ls_params = amazon_retriever._get_ls_params()
    assert ls_params == {"ls_retriever_name": "amazonknowledgebases"}


def test_get_relevant_documents(
    amazon_retriever: AmazonKnowledgeBasesRetriever, mock_client: MagicMock
) -> None:
    query: str = "test query"
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {"content": {"text": "result1"}, "metadata": {"key": "value1"}},
            {
                "content": {"text": "result2"},
                "metadata": {"key": "value2"},
                "score": 1,
                "location": "testLocation",
            },
            {"content": {"text": "result3"}},
        ]
    }
    documents: List[Document] = amazon_retriever._get_relevant_documents(
        query,
        run_manager=None,  # type: ignore
    )

    assert len(documents) == 3
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result1"
    assert documents[0].metadata == {"score": 0, "source_metadata": {"key": "value1"}}
    assert documents[1].page_content == "result2"
    assert documents[1].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
    }
    assert documents[2].page_content == "result3"
    assert documents[2].metadata == {"score": 0}
