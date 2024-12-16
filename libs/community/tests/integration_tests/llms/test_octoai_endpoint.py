"""Test OctoAI API wrapper."""

from langchain_community.llms.octoai_endpoint import OctoAIEndpoint


def test_octoai_endpoint_call() -> None:
    """Test valid call to OctoAI endpoint."""
    llm = OctoAIEndpoint()
    output = llm.invoke("Which state is Los Angeles in?")
    print(output)  # noqa: T201
    assert isinstance(output, str)
