from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, patch

import pytest
import responses
from langchain_core.documents import Document

from langchain_community.utilities.you import YouSearchAPIWrapper

TEST_ENDPOINT = "https://api.ydc-index.io"

# Mock you.com response for testing
MOCK_RESPONSE_RAW: Dict[str, List[Dict[str, Union[str, List[str]]]]] = {
    "hits": [
        {
            "description": "Test description",
            "snippets": ["yo", "bird up"],
            "thumbnail_url": "https://example.com/image.gif",
            "title": "Test title 1",
            "url": "https://example.com/article.html",
        },
        {
            "description": "Test description 2",
            "snippets": ["worst show", "on tv"],
            "thumbnail_url": "https://example.com/image2.gif",
            "title": "Test title 2",
            "url": "https://example.com/article2.html",
        },
    ]
}


def generate_parsed_metadata(num: Optional[int] = 0) -> Dict[Any, Any]:
    """generate metadata for testing"""
    if num is None:
        num = 0
    hit: Dict[str, Union[str, List[str]]] = MOCK_RESPONSE_RAW["hits"][num]
    return {
        "url": hit["url"],
        "thumbnail_url": hit["thumbnail_url"],
        "title": hit["title"],
        "description": hit["description"],
    }


def generate_parsed_output(num: Optional[int] = 0) -> List[Document]:
    """generate parsed output for testing"""
    if num is None:
        num = 0
    hit: Dict[str, Union[str, List[str]]] = MOCK_RESPONSE_RAW["hits"][num]
    output = []
    for snippit in hit["snippets"]:
        doc = Document(page_content=snippit, metadata=generate_parsed_metadata(num))
        output.append(doc)
    return output


# Mock results after parsing
MOCK_PARSED_OUTPUT = generate_parsed_output()
MOCK_PARSED_OUTPUT.extend(generate_parsed_output(1))
# Single-snippet
LIMITED_PARSED_OUTPUT = []
LIMITED_PARSED_OUTPUT.append(generate_parsed_output()[0])
LIMITED_PARSED_OUTPUT.append(generate_parsed_output(1)[0])

# copied from you api docs
NEWS_RESPONSE_RAW = {
    "news": {
        "results": [
            {
                "age": "18 hours ago",
                "breaking": True,
                "description": "Search on YDC for the news",
                "meta_url": {
                    "hostname": "www.reuters.com",
                    "netloc": "reuters.com",
                    "path": "› 2023  › 10  › 18  › politics  › inflation  › index.html",
                    "scheme": "https",
                },
                "page_age": "2 days",
                "page_fetched": "2023-10-12T23:00:00Z",
                "thumbnail": {"original": "https://reuters.com/news.jpg"},
                "title": "Breaking News about the World's Greatest Search Engine!",
                "type": "news",
                "url": "https://news.you.com",
            }
        ]
    }
}

NEWS_RESPONSE_PARSED = [
    Document(page_content=str(result["description"]), metadata=result)
    for result in NEWS_RESPONSE_RAW["news"]["results"]
]


@responses.activate
def test_raw_results() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
    )

    query = "Test query text"
    # ensure default endpoint_type
    you_wrapper = YouSearchAPIWrapper(endpoint_type="snippet", ydc_api_key="test")
    raw_results = you_wrapper.raw_results(query)
    expected_result = MOCK_RESPONSE_RAW
    assert raw_results == expected_result


@responses.activate
def test_raw_results_defaults() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
    )

    query = "Test query text"
    # ensure limit on number of docs returned
    you_wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    raw_results = you_wrapper.raw_results(query)
    expected_result = MOCK_RESPONSE_RAW
    assert raw_results == expected_result


@responses.activate
def test_raw_results_news() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/news", json=NEWS_RESPONSE_RAW, status=200
    )

    query = "Test news text"
    # ensure limit on number of docs returned
    you_wrapper = YouSearchAPIWrapper(endpoint_type="news", ydc_api_key="test")
    raw_results = you_wrapper.raw_results(query)
    expected_result = NEWS_RESPONSE_RAW
    assert raw_results == expected_result


@responses.activate
def test_results() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
    )

    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    results = you_wrapper.results(query)
    expected_result = MOCK_PARSED_OUTPUT
    assert results == expected_result


@responses.activate
def test_results_max_docs() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
    )

    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(k=2, ydc_api_key="test")
    results = you_wrapper.results(query)
    expected_result = generate_parsed_output()
    assert results == expected_result


@responses.activate
def test_results_limit_snippets() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/search", json=MOCK_RESPONSE_RAW, status=200
    )

    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(n_snippets_per_hit=1, ydc_api_key="test")
    results = you_wrapper.results(query)
    expected_result = LIMITED_PARSED_OUTPUT
    assert results == expected_result


@responses.activate
def test_results_news() -> None:
    responses.add(
        responses.GET, f"{TEST_ENDPOINT}/news", json=NEWS_RESPONSE_RAW, status=200
    )

    query = "Test news text"
    # ensure limit on number of docs returned
    you_wrapper = YouSearchAPIWrapper(endpoint_type="news", ydc_api_key="test")
    raw_results = you_wrapper.results(query)
    expected_result = NEWS_RESPONSE_PARSED
    assert raw_results == expected_result


@pytest.mark.asyncio
async def test_raw_results_async() -> None:
    instance = YouSearchAPIWrapper(ydc_api_key="test_api_key")

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
        results = await instance.raw_results_async("test query")
        assert results == MOCK_RESPONSE_RAW


@pytest.mark.asyncio
async def test_results_async() -> None:
    instance = YouSearchAPIWrapper(ydc_api_key="test_api_key")

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
        results = await instance.results_async("test query")
        assert results == MOCK_PARSED_OUTPUT


@pytest.mark.asyncio
async def test_results_news_async() -> None:
    instance = YouSearchAPIWrapper(endpoint_type="news", ydc_api_key="test_api_key")

    # Mock response object to simulate aiohttp response
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = (
        mock_response  # Make the context manager return itself
    )
    mock_response.__aexit__.return_value = None  # No value needed for exit
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=NEWS_RESPONSE_RAW)

    # Patch the aiohttp.ClientSession object
    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        results = await instance.results_async("test query")
        assert results == NEWS_RESPONSE_PARSED
