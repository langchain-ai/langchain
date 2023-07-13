"""Test the initialize module."""

from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.tools.base import tool
from tests.unit_tests.llms.fake_llm import FakeLLM


@tool
def my_tool(query: str) -> str:
    """A fake tool."""
    return "fake tool"


def test_initialize_agent_with_str_agent_type() -> None:
    """Test initialize_agent with a string."""
    fake_llm = FakeLLM()
    agent_executor = initialize_agent(
        [my_tool], fake_llm, "zero-shot-react-description"  # type: ignore
    )
    assert agent_executor.agent._agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION
    assert isinstance(agent_executor.tags, list)
    assert "zero-shot-react-description" in agent_executor.tags
