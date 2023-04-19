"""Integration test for Wikipedia API Wrapper."""
import pytest

from langchain.utilities import ArxivAPIWrapper


@pytest.fixture
def api_client():
    return ArxivAPIWrapper()


def test_call(api_client) -> None:
    """Test that ArxivAPIWrapper returns correct answer"""

    output = api_client.run("1605.08386")
    assert len(output) == 1
    assert "Heat-bath random walks with Markov bases" in output[0]


def test_several_docs(api_client) -> None:
    """Test that ArxivAPIWrapper returns several docs"""

    docs = api_client.run("Caprice Stanley")
    assert len(docs) == 3
    assert any("On Mixing Behavior of a Family of Random Walks" for doc in docs)


def test_no_result_call(api_client) -> None:
    """Test that call gives no result."""

    docs = api_client.run("1605.08386WWW")
    assert not docs
