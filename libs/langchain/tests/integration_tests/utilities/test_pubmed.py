"""Integration test for PubMed API Wrapper."""
from typing import Any, List

import pytest

from langchain.agents.load_tools import load_tools
from langchain.schema import Document
from langchain.tools import PubmedQueryRun
from langchain.tools.base import BaseTool
from langchain.utilities import PubMedAPIWrapper

xmltodict = pytest.importorskip("xmltodict")


@pytest.fixture
def api_client() -> PubMedAPIWrapper:
    return PubMedAPIWrapper()


def test_run_success(api_client: PubMedAPIWrapper) -> None:
    """Test that returns the correct answer"""

    output = api_client.run("chatgpt")
    assert "Performance of ChatGPT on the Situational Judgement Test-A" in output
    assert len(output) == api_client.doc_content_chars_max


def test_run_returns_no_result(api_client: PubMedAPIWrapper) -> None:
    """Test that gives no result."""

    output = api_client.run("1605.08386WWW")
    assert "No good PubMed Result was found" == output


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.metadata
        assert set(doc.metadata) == {
            "Copyright Information",
            "uid",
            "Title",
            "Published",
        }


def test_load_success(api_client: PubMedAPIWrapper) -> None:
    """Test that returns one document"""

    docs = api_client.load_docs("chatgpt")
    assert len(docs) == api_client.top_k_results == 3
    assert_docs(docs)


def test_load_returns_no_result(api_client: PubMedAPIWrapper) -> None:
    """Test that returns no docs"""

    docs = api_client.load_docs("1605.08386WWW")
    assert len(docs) == 0


def test_load_returns_limited_docs() -> None:
    """Test that returns several docs"""
    expected_docs = 2
    api_client = PubMedAPIWrapper(top_k_results=expected_docs)
    docs = api_client.load_docs("ChatGPT")
    assert len(docs) == expected_docs
    assert_docs(docs)


def test_load_returns_full_set_of_metadata() -> None:
    """Test that returns several docs"""
    api_client = PubMedAPIWrapper(load_max_docs=1, load_all_available_meta=True)
    docs = api_client.load_docs("ChatGPT")
    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata
        assert set(doc.metadata).issuperset(
            {"Copyright Information", "Published", "Title", "uid"}
        )


def _load_pubmed_from_universal_entry(**kwargs: Any) -> BaseTool:
    tools = load_tools(["pubmed"], **kwargs)
    assert len(tools) == 1, "loaded more than 1 tool"
    return tools[0]


def test_load_pupmed_from_universal_entry() -> None:
    pubmed_tool = _load_pubmed_from_universal_entry()
    output = pubmed_tool("chatgpt")
    assert "Performance of ChatGPT on the Situational Judgement Test-A" in output


def test_load_pupmed_from_universal_entry_with_params() -> None:
    params = {
        "top_k_results": 1,
    }
    pubmed_tool = _load_pubmed_from_universal_entry(**params)
    assert isinstance(pubmed_tool, PubmedQueryRun)
    wp = pubmed_tool.api_wrapper
    assert wp.top_k_results == 1, "failed to assert top_k_results"
