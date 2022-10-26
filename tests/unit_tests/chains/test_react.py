"""Unit tests for ReAct."""

from typing import List, Optional
from unittest.mock import Mock, patch

import pytest

from langchain.chains.llm import LLMChain
from langchain.chains.react.base import (
    PageWithLookups,
    ReActChain,
    predict_until_observation,
)
from langchain.llms.base import LLM
from langchain.prompt import Prompt

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?

Made in 2022."""

_FAKE_PROMPT = Prompt(input_variables=["input"], template="{input}")


class FakeListLLM(LLM):
    """Fake LLM for testing that outputs elements of a list."""

    def __init__(self, responses: List[str]):
        """Initialize with list of responses."""
        self.responses = responses
        self.i = -1

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Increment counter, and then return response in that index."""
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


def test_page_with_lookups_dont_exist() -> None:
    """Test lookup on term that doesn't exist in the page."""
    page = PageWithLookups(page_content=_PAGE_CONTENT)

    # Start with lookup on "harrison".
    output = page.lookup("harrison")
    assert output == "No Results"


def test_page_with_lookups_too_many() -> None:
    """Test lookup on term too many times."""
    page = PageWithLookups(page_content=_PAGE_CONTENT)

    # Start with lookup on "framework".
    output = page.lookup("framework")
    assert output == "(Result 1/1) It is a really cool framework."

    # Now try again, should be exhausted.
    output = page.lookup("framework")
    assert output == "No More Results"


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
        "I should probably search\nAction 1: Search[langchain]",
        "I should probably lookup\nAction 2: Lookup[made]",
        "Ah okay now I know the answer\nAction 3: Finish[2022]",
    ]
    fake_llm = FakeListLLM(responses)
    react_chain = ReActChain(llm=fake_llm)
    inputs = {"question": "when was langchain made"}
    fake_return = Mock()
    fake_return.content = _PAGE_CONTENT
    with patch("wikipedia.page", return_value=fake_return):
        output = react_chain(inputs)
    assert output["answer"] == "2022"
    expected_full_output = (
        "when was langchain made\n"
        "Thought 1:I should probably search\n"
        "Action 1: Search[langchain]\n"
        "Observation 1: This is a page about LangChain.\n"
        "Thought 2:I should probably lookup\n"
        "Action 2: Lookup[made]\n"
        "Observation 2: (Result 1/1) Made in 2022.\n"
        "Thought 3:Ah okay now I know the answer\n"
        "Action 3: Finish[2022]"
    )
    assert output["full_logic"] == expected_full_output


def test_react_chain_bad_action() -> None:
    """Test react chain when bad action given."""
    responses = [
        "I should probably search\nAction 1: BadAction[langchain]",
    ]
    fake_llm = FakeListLLM(responses)
    react_chain = ReActChain(llm=fake_llm)
    with pytest.raises(ValueError):
        react_chain.run("when was langchain made")
