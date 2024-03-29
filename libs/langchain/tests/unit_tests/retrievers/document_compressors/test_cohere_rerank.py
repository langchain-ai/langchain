import os

import pytest
from pytest_mock import MockerFixture

from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document

os.environ["COHERE_API_KEY"] = "foo"


@pytest.mark.requires("cohere")
def test_init() -> None:
    CohereRerank()

    CohereRerank(
        top_n=5, model="rerank-english_v2.0", cohere_api_key="foo", user_agent="bar"
    )


@pytest.mark.requires("cohere")
def test_rerank(mocker: MockerFixture) -> None:
    mock_client = mocker.MagicMock()
    mock_result = mocker.MagicMock()
    mock_result.results = [
        mocker.MagicMock(index=0, relevance_score=0.8),
        mocker.MagicMock(index=1, relevance_score=0.6),
    ]
    mock_client.rerank.return_value = mock_result

    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"

    mocker.patch("cohere.Client", return_value=mock_client)

    reranker = CohereRerank(cohere_api_key="foo")
    results = reranker.rerank(test_documents, test_query)

    mock_client.rerank.assert_called_once_with(
        query=test_query,
        documents=[doc.page_content for doc in test_documents],
        model="rerank-english-v2.0",
        top_n=3,
        max_chunks_per_doc=None,
    )
    assert results == [
        {"index": 0, "relevance_score": 0.8},
        {"index": 1, "relevance_score": 0.6},
    ]
