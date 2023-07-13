"""Test Bittensor Validator Endpoint wrapper."""

from langchain.llms.bittensor import NIBittensorLLM


def test_bittensor_call() -> None:
    """Test valid call to validator endpoint."""
    llm = NIBittensorLLM(system="Your task is to answer user prompt.")
    output = llm("Say foo:")
    assert isinstance(output, str)
