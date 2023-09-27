"""Test Baseten API wrapper."""
import os

from langchain.llms.baseten import Baseten


def test_baseten_call() -> None:
    """Test valid call to Baseten."""
    import baseten

    baseten.login(os.environ["BASETEN_API_KEY"])
    llm = Baseten(model=os.environ["BASETEN_MODEL_ID"])
    output = llm("Say foo:")
    assert isinstance(output, str)
