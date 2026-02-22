from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from langchain_perplexity import PerplexitySearchRetriever


def test_search_retriever_initialization() -> None:
    retriever = PerplexitySearchRetriever(pplx_api_key="test")
    assert retriever.pplx_api_key.get_secret_value() == "test"
    assert retriever.k == 10


def test_search_retriever_get_relevant_documents(mocker: MockerFixture) -> None:
    retriever = PerplexitySearchRetriever(pplx_api_key="test")

    mock_result = MagicMock()
    mock_result.title = "Test Title"
    mock_result.url = "http://test.com"
    mock_result.snippet = "Test snippet"
    mock_result.date = "2023-01-01"
    mock_result.last_updated = "2023-01-02"

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mock_create = MagicMock(return_value=mock_response)
    mocker.patch.object(retriever.client.search, "create", mock_create)

    docs = retriever.invoke("query")

    assert len(docs) == 1
    assert docs[0].page_content == "Test snippet"
    assert docs[0].metadata["title"] == "Test Title"
    assert docs[0].metadata["url"] == "http://test.com"

    mock_create.assert_called_once_with(
        query="query",
        max_results=10,
        max_tokens=25000,
        max_tokens_per_page=1024,
    )
