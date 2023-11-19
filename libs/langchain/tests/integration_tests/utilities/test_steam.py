"""Integration test for Steam Web API Wrapper."""

import pytest

# from langchain.utilities import SteamWebAPIWrapper
from langchain.utilities.steam import SteamWebAPIWrapper


@pytest.fixture
def api_client() -> SteamWebAPIWrapper:
    return SteamWebAPIWrapper()


def test_run_llm(api_client: SteamWebAPIWrapper) -> None:
    output = api_client.run("LLM")
    assert "large language model" in output
