from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from langchain_perplexity import PerplexitySearchResults


def test_search_tool_run(mocker: MockerFixture) -> None:
    tool = PerplexitySearchResults(pplx_api_key="test")

    mock_result = MagicMock()
    mock_result.title = "Test Title"
    mock_result.url = "http://test.com"
    mock_result.snippet = "Test snippet"
    mock_result.date = "2023-01-01"
    mock_result.last_updated = "2023-01-02"

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mock_create = MagicMock(return_value=mock_response)
    mocker.patch.object(tool.client.search, "create", mock_create)

    result = tool.invoke("query")

    # result should be a list of dicts (converted by tool) or str if string output
    # By default, tool.invoke returns the output of _run.
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["title"] == "Test Title"

    mock_create.assert_called_once_with(
        query="query",
        max_results=10,
    )
