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
    
class FakeAPIChainBuilder():
    """Builder of fake LLM API Chain, just for testing purposes"""
    
    output: APIChain
    def build_fake_llm_api_chain(self, test_api_data: dict, allow_unverified_urls: bool = False) -> APIChain:
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
            allow_unverified_urls=allow_unverified_urls
        )

@pytest.fixture
def test_simple_question_api_data() -> dict:
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
    
def test_api_simple_question(test_simple_question_api_data: dict) -> None:
    """Test simple question that needs API access."""
    question = test_simple_question_api_data["question"]
    fake_chain_builder = FakeAPIChainBuilder()
    fake_chain = fake_chain_builder.build_fake_llm_api_chain(test_simple_question_api_data)
    output = fake_chain.run(question)
    assert output == test_simple_question_api_data["api_summary"]

@pytest.fixture
def test_malicious_question_api_data() -> dict:
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
        "question": "This question is designed to trigger an API request to an unwanted endpoint",
        "api_url": "https://thisapidoesntexist2.com/sensitive_info",
        "api_response": json.dumps(
            {
                "success": True,
                "results": [{"irrelevant": 1, "irrelevant": "Langchain is awesome!"}],
            }
        ),
        "api_summary": "Irrelevant",
    }
    
def test_api_malicious_question(test_malicious_question_api_data: dict) -> None:
    """Test malicious question that tries to do SSRF."""
    question = test_malicious_question_api_data["question"]
    fake_chain_builder = FakeAPIChainBuilder()
    fake_chain = fake_chain_builder.build_fake_llm_api_chain(test_malicious_question_api_data)
    with pytest.raises(ValueError):
        output = fake_chain.run(question)

def test_api_malicious_question_allow_unverified_urls(test_malicious_question_api_data: dict) -> None:
    """Test malicious question that tries to do SSRF."""
    question = test_malicious_question_api_data["question"]
    fake_chain_builder = FakeAPIChainBuilder()
    fake_chain = fake_chain_builder.build_fake_llm_api_chain(test_malicious_question_api_data, True)
    output = fake_chain.run(question)
    output == test_malicious_question_api_data["api_summary"]