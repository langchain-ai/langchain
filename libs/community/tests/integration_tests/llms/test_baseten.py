"""Test Baseten API wrapper."""

import os

from langchain_community.llms.baseten import Baseten

# This test requires valid BASETEN_MODEL_ID and BASETEN_API_KEY environment variables


def test_baseten_call() -> None:
    """Test valid call to Baseten."""
    llm = Baseten(model=os.environ["BASETEN_MODEL_ID"])  # type: ignore[call-arg]
    output = llm.invoke("Test prompt, please respond.")
    assert isinstance(output, str)
