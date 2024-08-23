from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_community.tools.you import YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper

from ..utilities.test_you import (
    LIMITED_PARSED_OUTPUT,
    MOCK_PARSED_OUTPUT,
    MOCK_RESPONSE_RAW,
    NEWS_RESPONSE_PARSED,
    NEWS_RESPONSE_RAW,
    TEST_ENDPOINT,
)


class TestYouSearchTool:
    @responses.activate
    def test_invoke(self) -> None:
        responses.add(
            responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
        )
        query = "Test query text"
        you_tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))  # type: ignore[call-arg]
        results = you_tool.invoke(query)
        expected_result = MOCK_PARSED_OUTPUT
        assert results == expected_result

    @responses.activate
    def test_invoke_max_docs(self) -> None:
        responses.add(
            responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
        )
        query = "Test query text"
        you_tool = YouSearchTool(  # type: ignore[call-arg]
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", k=2)
        )
        results = you_tool.invoke(query)
        expected_result = [MOCK_PARSED_OUTPUT[0], MOCK_PARSED_OUTPUT[1]]
        assert results == expected_result

    @responses.activate
    def test_invoke_limit_snippets(self) -> None:
        responses.add(
            responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
        )
        query = "Test query text"
        you_tool = YouSearchTool(  # type: ignore[call-arg]
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", n_snippets_per_hit=1)
        )
        results = you_tool.invoke(query)
        expected_result = LIMITED_PARSED_OUTPUT
        assert results == expected_result

    @responses.activate
    def test_invoke_news(self) -> None:
        responses.add(
            responses.GET, f"{TEST_ENDPOINT}/news", json=NEWS_RESPONSE_RAW, status=200
        )

        query = "Test news text"
        you_tool = YouSearchTool(  # type: ignore[call-arg]
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", endpoint_type="news")
        )
        results = you_tool.invoke(query)
        expected_result = NEWS_RESPONSE_PARSED
        assert results == expected_result

    @pytest.mark.asyncio
    async def test_ainvoke(self) -> None:
        you_tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))  # type: ignore[call-arg]

        # Mock response object to simulate aiohttp response
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = (
            mock_response  # Make the context manager return itself
        )
        mock_response.__aexit__.return_value = None  # No value needed for exit
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MOCK_RESPONSE_RAW)

        # Patch the aiohttp.ClientSession object
        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            results = await you_tool.ainvoke("test query")
            assert results == MOCK_PARSED_OUTPUT
