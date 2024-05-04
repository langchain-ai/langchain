"""Test Bittensor Validator Endpoint wrapper."""

from langchain_community.llms import NIBittensorLLM


def test_bittensor_call() -> None:
    """Test valid call to validator endpoint."""
    llm = NIBittensorLLM(system_prompt="Your task is to answer user prompt.")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
