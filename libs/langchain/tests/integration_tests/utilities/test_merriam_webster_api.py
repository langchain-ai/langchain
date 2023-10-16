"""Integration test for Wikipedia API Wrapper."""
from typing import List

import pytest

from langchain.schema import Document
from langchain.utilities import MerriamWebsterAPIWrapper


@pytest.fixture
def api_client() -> MerriamWebsterAPIWrapper:
    return MerriamWebsterAPIWrapper()


def test_run_not_iplemented(api_client: MerriamWebsterAPIWrapper) -> None:
    output = api_client.run("Foo")
    assert "Merriam-Webster API is not implemented yet" == output
