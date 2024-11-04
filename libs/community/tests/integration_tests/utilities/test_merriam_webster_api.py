"""Integration test for Merriam Webster API Wrapper."""

import pytest

from langchain_community.utilities.merriam_webster import MerriamWebsterAPIWrapper


@pytest.fixture
def api_client() -> MerriamWebsterAPIWrapper:
    return MerriamWebsterAPIWrapper()


def test_call(api_client: MerriamWebsterAPIWrapper) -> None:
    """Test that call gives correct answer."""
    output = api_client.run("LLM")
    assert "large language model" in output


def test_call_no_result(api_client: MerriamWebsterAPIWrapper) -> None:
    """Test that non-existent words return proper result."""
    output = api_client.run("NO_RESULT_NO_RESULT_NO_RESULT")
    assert "No Merriam-Webster definition was found for query" in output


def test_call_alternatives(api_client: MerriamWebsterAPIWrapper) -> None:
    """
    Test that non-existent queries that are close to an
    existing definition return proper result.
    """
    output = api_client.run("It's raining cats and dogs")
    assert "No Merriam-Webster definition was found for query" in output
    assert "You can try one of the following alternative queries" in output
    assert "raining cats and dogs" in output
