"""Integration test for Wikipedia API Wrapper."""
import pytest

from langchain.utilities import MerriamWebsterAPIWrapper


@pytest.fixture
def api_client() -> MerriamWebsterAPIWrapper:
    return MerriamWebsterAPIWrapper()


def test_run_llm(api_client: MerriamWebsterAPIWrapper) -> None:
    output = api_client.run("LLM")
    assert "large language model" in output
