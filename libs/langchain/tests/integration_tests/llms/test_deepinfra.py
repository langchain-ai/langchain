"""Test DeepInfra API wrapper."""
import pytest

from langchain.llms.deepinfra import DeepInfra


def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    llm = DeepInfra(model_id="google/flan-t5-small")
    output = llm("What is 2 + 2?")
    assert isinstance(output, str)

@pytest.mark.asyncio
async def test_deepinfra_acall() -> None:
    llm = DeepInfra(model_id="google/flan-t5-small")
    output = await llm.apredict("Say foo:")
    assert llm._llm_type == "deepinfra"
    assert isinstance(output, str)
