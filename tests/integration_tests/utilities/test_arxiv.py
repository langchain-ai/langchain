"""Integration test for Arxiv API Wrapper."""
import pytest

from langchain.utilities import ArxivAPIWrapper


@pytest.fixture
def api_client() -> ArxivAPIWrapper:
    return ArxivAPIWrapper()


def test_call(api_client: ArxivAPIWrapper) -> None:
    """Test that ArxivAPIWrapper returns correct answer"""

    output = api_client.run("1605.08386")
    assert "Heat-bath random walks with Markov bases" in output


def test_several_docs(api_client: ArxivAPIWrapper) -> None:
    """Test that ArxivAPIWrapper returns several docs"""

    output = api_client.run("Caprice Stanley")
    assert "On Mixing Behavior of a Family of Random Walks" in output


def test_no_result_call(api_client: ArxivAPIWrapper) -> None:
    """Test that call gives no result."""

    output = api_client.run("1605.08386WWW")
    assert "No good Arxiv Result was found" == output
