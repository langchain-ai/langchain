"""Test LLM Math functionality."""

import json
from typing import Any

import pytest
from langchain_community.utilities.requests import TextRequestsWrapper

from langchain.chains.api.base import APIChain
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.llm import LLMChain
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeRequestsChain(TextRequestsWrapper):
    """Fake requests chain just for testing purposes."""

    output: str

    def get(self, url: str, **kwargs: Any) -> str:
        """Just return the specified output."""
        return self.output


def get_test_api_data() -> dict:
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


def get_api_chain(**kwargs: Any) -> APIChain:
    """Fake LLM API chain for testing."""
    data = get_test_api_data()
    test_api_docs = data["api_docs"]
    test_question = data["question"]
    test_url = data["api_url"]
    test_api_response = data["api_response"]
    test_api_summary = data["api_summary"]

    api_url_query_prompt = API_URL_PROMPT.format(
        api_docs=test_api_docs, question=test_question
    )
    api_response_prompt = API_RESPONSE_PROMPT.format(
        api_docs=test_api_docs,
        question=test_question,
        api_url=test_url,
        api_response=test_api_response,
    )
    queries = {api_url_query_prompt: test_url, api_response_prompt: test_api_summary}
    fake_llm = FakeLLM(queries=queries)
    api_request_chain = LLMChain(llm=fake_llm, prompt=API_URL_PROMPT)
    api_answer_chain = LLMChain(llm=fake_llm, prompt=API_RESPONSE_PROMPT)
    requests_wrapper = FakeRequestsChain(output=test_api_response)
    return APIChain(
        api_request_chain=api_request_chain,
        api_answer_chain=api_answer_chain,
        requests_wrapper=requests_wrapper,
        api_docs=test_api_docs,
        **kwargs,
    )


def test_api_question() -> None:
    """Test simple question that needs API access."""
    with pytest.raises(ValueError):
        get_api_chain()
    with pytest.raises(ValueError):
        get_api_chain(limit_to_domains=tuple())

    # All domains allowed (not advised)
    api_chain = get_api_chain(limit_to_domains=None)
    data = get_test_api_data()
    assert api_chain.run(data["question"]) == data["api_summary"]

    # Use a domain that's allowed
    api_chain = get_api_chain(
        limit_to_domains=["https://thisapidoesntexist.com/api/notes?q=langchain"]
    )
    # Attempts to make a request against a domain that's not allowed
    assert api_chain.run(data["question"]) == data["api_summary"]

    # Use domains that are not valid
    api_chain = get_api_chain(limit_to_domains=["h", "*"])
    with pytest.raises(ValueError):
        # Attempts to make a request against a domain that's not allowed
        assert api_chain.run(data["question"]) == data["api_summary"]
