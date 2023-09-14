"""Test DeepInfra API wrapper."""

from langchain.llms.deepinfra import DeepInfra


def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    llm = DeepInfra(model_id="google/flan-t5-small")
    output = llm("What is 2 + 2?")
    assert isinstance(output, str)
