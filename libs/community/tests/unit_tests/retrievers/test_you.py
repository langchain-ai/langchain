
import responses

from langchain_community.retrievers.you import YouRetriever

from ..utilities.test_you import (
    MOCK_OUTPUT_PARSED,
    MOCK_RESPONSE_RAW,
    TEST_ENDPOINT,
)


class TestYouRetriever:
    @responses.activate
    def test_get_relevant_documents(self) -> None:
        responses.add(responses.GET, TEST_ENDPOINT, json=MOCK_RESPONSE_RAW, status=200)
        query = "Test query text"
        you_wrapper = YouRetriever(num_web_results=1)
        raw_results = you_wrapper.get_relevant_documents(query)
        expected_result = MOCK_OUTPUT_PARSED
        assert raw_results == expected_result

    @responses.activate
    def test_invoke(self) -> None:
        responses.add(responses.GET, TEST_ENDPOINT, json=MOCK_RESPONSE_RAW, status=200)
        query = "Test query text"
        you_wrapper = YouRetriever(num_web_results=1)
        raw_results = you_wrapper.invoke(query)
        expected_result = MOCK_OUTPUT_PARSED
        assert raw_results == expected_result
