"""Unit tests for ReAct."""

from langchain.chains.react.base import predict_until_observation, PageWithLookups, ReActChain
import pytest
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.prompt import Prompt
from typing import List, Optional
from unittest.mock import patch

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?"""

_FAKE_PROMPT = Prompt(input_variables=["input"], template="{input}")


class FakeListLLM(LLM):
    """Fake LLM for testing that outputs elements of a list."""

    def __init__(self, responses: List[str]):
        """Initialize with list of responses."""
        self.responses = responses
        self.i = -1

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.i += 1
        return self.responses[self.i]


def test_page_with_lookups_summary() -> None:
    """Test that we extract the summary okay."""
    page = PageWithLookups(page_content=_PAGE_CONTENT)
    assert page.summary == "This is a page about LangChain."


def test_page_with_lookups_lookup() -> None:
    """Test that can lookup things okay."""
    page = PageWithLookups(page_content=_PAGE_CONTENT)

    # Start with lookup on "LangChain".
    output = page.lookup("LangChain")
    assert output == "(Result 1/2) This is a page about LangChain."

    # Now switch to looking up "framework".
    output = page.lookup("framework")
    assert output == "(Result 1/1) It is a really cool framework."

    # Now switch back to looking up "LangChain", should reset.
    output = page.lookup("LangChain")
    assert output == "(Result 1/2) This is a page about LangChain."

    # Lookup "LangChain" again, should go to the next mention.
    output = page.lookup("LangChain")
    assert output == "(Result 2/2) What isn't there to love about langchain?"




def test_predict_until_observation_normal() -> None:
    """Test predict_until_observation when observation is made normally."""
    outputs = ["foo\nAction 1: search[foo]"]
    fake_llm = FakeListLLM(outputs)
    fake_llm_chain = LLMChain(llm=fake_llm, prompt=_FAKE_PROMPT)
    ret_text, action, directive = predict_until_observation(fake_llm_chain, "", 1)
    assert ret_text == outputs[0]
    assert action == "search"
    assert directive == "foo"

def test_predict_until_observation_repeat() -> None:
    """Test when no action is generated initially."""
    outputs = ["foo", " search[foo]"]
    fake_llm = FakeListLLM(outputs)
    fake_llm_chain = LLMChain(llm=fake_llm, prompt=_FAKE_PROMPT)
    ret_text, action, directive = predict_until_observation(fake_llm_chain, "", 1)
    assert ret_text == "foo\nAction 1: search[foo]"
    assert action == "search"
    assert directive == "foo"


def test_predict_until_observation_error() -> None:
    """Test handling of generation of text that cannot be parsed."""
    outputs = ["foo\nAction 1: foo"]
    fake_llm = FakeListLLM(outputs)
    fake_llm_chain = LLMChain(llm=fake_llm, prompt=_FAKE_PROMPT)
    with pytest.raises(ValueError):
        predict_until_observation(fake_llm_chain, "", 1)


def test_react_chain() -> None:
    """Test react chain."""
    responses = [
        "I should probabaly search\nAction 1: Search[langchain]",
        "I should probably"
    ]
    fake_llm = FakeListLLM(responses)
    react_chain = ReActChain(llm=fake_llm)
    inputs = {"question": "when was langchain made"}
    with patch("wikipedia.page", return_value=_PAGE_CONTENT):
        react_chain(inputs)