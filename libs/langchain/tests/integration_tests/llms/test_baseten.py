"""Test Baseten API wrapper."""
import os

import baseten
import pytest

from langchain.llms.baseten import Baseten


@pytest.mark.requires(baseten)
def test_baseten_call() -> None:
    """Test valid call to Baseten."""
    baseten.login(os.environ["BASETEN_API_KEY"])
    llm = Baseten(model=os.environ["BASETEN_MODEL_ID"])
    output = llm("Say foo:")
    assert isinstance(output, str)
