"""Test OpenLLM API wrapper."""

import pytest

from langchain_community.llms.openllm import OpenLLM


@pytest.mark.scheduled
def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenLLM()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)
