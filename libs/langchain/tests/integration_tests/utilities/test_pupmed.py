"""Integration test for PubMed API Wrapper."""
from typing import Any, List

import pytest

from langchain.agents.load_tools import load_tools
from langchain.schema import Document
from langchain.tools.base import BaseTool
from langchain.utilities import PubMedAPIWrapper


@pytest.fixture
def api_client() -> PubMedAPIWrapper:
    return PubMedAPIWrapper()


def test_run_success(api_client: PubMedAPIWrapper) -> None:
    """Test that returns the correct answer"""

    output = api_client.run("1605.08386")
    assert "Heat-bath random walks with Markov bases" in output


def test_run_returns_several_docs(api_client: PubMedAPIWrapper) -> None:
    """Test that returns several docs"""

    output = api_client.run("Caprice Stanley")
    assert "On Mixing Behavior of a Family of Random Walks" in output


def test_run_returns_no_result(api_client: PubMedAPIWrapper) -> None:
    """Test that gives no result."""

    output = api_client.run("1605.08386WWW")
    assert "No good PubMed Result was found" == output


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {"Published", "Title", "Authors", "Summary"}


def test_load_success(api_client: PubMedAPIWrapper) -> None:
    """Test that returns one document"""

    docs = api_client.load_docs("1605.08386")
    assert len(docs) == 1
    assert_docs(docs)


def test_load_returns_no_result(api_client: PubMedAPIWrapper) -> None:
    """Test that returns no docs"""

    docs = api_client.load("1605.08386WWW")
    assert len(docs) == 0


def test_load_returns_limited_docs() -> None:
    """Test that returns several docs"""
    expected_docs = 2
    api_client = PubMedAPIWrapper(load_max_docs=expected_docs)
    docs = api_client.load_docs("ChatGPT")
    assert len(docs) == expected_docs
    assert_docs(docs)


def test_load_returns_full_set_of_metadata() -> None:
    """Test that returns several docs"""
    api_client = PubMedAPIWrapper(load_max_docs=1, load_all_available_meta=True)
    docs = api_client.load_docs("ChatGPT")
    assert len(docs) == 1
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata).issuperset(
            {"Published", "Title", "Authors", "Summary"}
        )
        print(doc.metadata)
        assert len(set(doc.metadata)) > 4


def _load_pubmed_from_universal_entry(**kwargs: Any) -> BaseTool:
    tools = load_tools(["pupmed"], **kwargs)
    assert len(tools) == 1, "loaded more than 1 tool"
    return tools[0]


def test_load_pupmed_from_universal_entry() -> None:
    pupmed_tool = _load_pubmed_from_universal_entry()
    output = pupmed_tool("Caprice Stanley")
    assert (
        "On Mixing Behavior of a Family of Random Walks" in output
    ), "failed to fetch a valid result"


def test_load_pupmed_from_universal_entry_with_params() -> None:
    params = {
        "top_k_results": 1,
        "load_max_docs": 10,
        "load_all_available_meta": True,
    }
    pupmed_tool = _load_pubmed_from_universal_entry(**params)
    assert isinstance(pupmed_tool, PubMedAPIWrapper)
    wp = pupmed_tool.api_wrapper
    assert wp.top_k_results == 1, "failed to assert top_k_results"
    assert wp.load_max_docs == 10, "failed to assert load_max_docs"
    assert (
        wp.load_all_available_meta is True
    ), "failed to assert load_all_available_meta"
