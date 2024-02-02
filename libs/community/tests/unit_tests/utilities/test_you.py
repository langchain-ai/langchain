import responses
from langchain_core.documents import Document

from langchain_community.utilities.you import YouSearchAPIWrapper

TEST_ENDPOINT = "https://api.ydc-index.io/search"

# Mock you.com response for testing
MOCK_RESPONSE_RAW = {
    "hits": [
        {
            "description": "Test description",
            "snippets": ["yo", "bird up"],
            "thumbnail_url": "https://example.com/image.gif",
            "title": "Test title 1",
            "url": "https://example.com/article.html",
        }
    ],
    "latency": 0.16670823097229004,
}

# Mock of parsed metadata
MOCK_PARSED_METADATA = {
    "url": MOCK_RESPONSE_RAW["hits"][0]["url"],
    "thumbnail_url": MOCK_RESPONSE_RAW["hits"][0]["thumbnail_url"],
    "title": MOCK_RESPONSE_RAW["hits"][0]["title"],
    "description": MOCK_RESPONSE_RAW["hits"][0]["description"],
}

# Mock results after parsing
MOCK_OUTPUT_PARSED = [
    Document(
        page_content=MOCK_RESPONSE_RAW["hits"][0]["snippets"][0],
        metadata=MOCK_PARSED_METADATA,
    ),
    Document(
        page_content=MOCK_RESPONSE_RAW["hits"][0]["snippets"][1],
        metadata=MOCK_PARSED_METADATA,
    ),
]


@responses.activate
def test_raw_results() -> None:
    responses.add(responses.GET, TEST_ENDPOINT, json=MOCK_RESPONSE_RAW, status=200)

    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper()
    raw_results = you_wrapper.raw_results(query, num_web_results=1)
    expected_result = MOCK_RESPONSE_RAW
    assert raw_results == expected_result


@responses.activate
def test_results() -> None:
    responses.add(responses.GET, TEST_ENDPOINT, json=MOCK_RESPONSE_RAW, status=200)

    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper()
    results = you_wrapper.results(query, num_web_results=1)
    expected_result = MOCK_OUTPUT_PARSED
    assert results == expected_result
