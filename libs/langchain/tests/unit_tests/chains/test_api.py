"""Test LLM Math functionality."""

import json
from typing import Any

import pytest

from langchain import LLMChain
from langchain.chains.api.base import APIChain
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.requests import TextRequestsWrapper
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeRequestsChain(TextRequestsWrapper):
    """Fake requests chain just for testing purposes."""

    output: str

    def get(self, url: str, **kwargs: Any) -> str:
        """Just return the specified output."""
        return self.output


@pytest.fixture
def test_api_data() -> dict:
    """Fake api data to use for testing."""
    api_docs = """
    This API endpoint will search the notes for a user.

    Endpoint: https://thisapidoesntexist.com
    GET /api/notes

    Query parameters:
    q | string | The search term for notes
    """
    return {
        "api_docs": api_docs,
        "question": "Search for notes containing langchain",
        "api_url": "https://thisapidoesntexist.com/api/notes?q=langchain",
        "api_response": json.dumps(
            {
                "success": True,
                "results": [{"id": 1, "content": "Langchain is awesome!"}],
            }
        ),
        "api_summary": "There is 1 note about langchain.",
    }


@pytest.fixture
def fake_llm_api_chain(test_api_data: dict) -> APIChain:
    """Fake LLM API chain for testing."""
    TEST_API_DOCS = test_api_data["api_docs"]
    TEST_QUESTION = test_api_data["question"]
    TEST_URL = test_api_data["api_url"]
    TEST_API_RESPONSE = test_api_data["api_response"]
    TEST_API_SUMMARY = test_api_data["api_summary"]

    api_url_query_prompt = API_URL_PROMPT.format(
        api_docs=TEST_API_DOCS, question=TEST_QUESTION
    )
    api_response_prompt = API_RESPONSE_PROMPT.format(
        api_docs=TEST_API_DOCS,
        question=TEST_QUESTION,
        api_url=TEST_URL,
        api_response=TEST_API_RESPONSE,
    )
    queries = {api_url_query_prompt: TEST_URL, api_response_prompt: TEST_API_SUMMARY}
    fake_llm = FakeLLM(queries=queries)
    api_request_chain = LLMChain(llm=fake_llm, prompt=API_URL_PROMPT)
    api_answer_chain = LLMChain(llm=fake_llm, prompt=API_RESPONSE_PROMPT)
    requests_wrapper = FakeRequestsChain(output=TEST_API_RESPONSE)
    return APIChain(
        api_request_chain=api_request_chain,
        api_answer_chain=api_answer_chain,
        requests_wrapper=requests_wrapper,
        api_docs=TEST_API_DOCS,
    )


def test_api_question(fake_llm_api_chain: APIChain, test_api_data: dict) -> None:
    """Test simple question that needs API access."""
    question = test_api_data["question"]
    output = fake_llm_api_chain.run(question)
    assert output == test_api_data["api_summary"]
