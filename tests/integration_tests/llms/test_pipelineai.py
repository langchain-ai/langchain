"""Test Pipeline Cloud API wrapper."""

from langchain.llms.pipelineai import PipelineAI


def test_pipelineai_call() -> None:
    """Test valid call to Pipeline Cloud."""
    llm = PipelineAI()
    output = llm("Say foo:")
    assert isinstance(output, str)
