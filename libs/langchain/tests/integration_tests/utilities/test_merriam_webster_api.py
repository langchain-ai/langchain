"""Integration test for Merriam Webster API Wrapper."""
import pytest

from langchain.utilities.merriam_webster import MerriamWebsterAPIWrapper


@pytest.fixture
def api_client() -> MerriamWebsterAPIWrapper:
    return MerriamWebsterAPIWrapper()


def test_call(api_client: MerriamWebsterAPIWrapper) -> None:
    output = api_client.run("LLM")
    assert "large language model" in output

def test_call_no_result(api_client: MerriamWebsterAPIWrapper) -> None:
    output = api_client.run("NO_RESULT_NO_RESULT_NO_RESULT")
    assert "No Merriam-Webster definition was found for query" in output
