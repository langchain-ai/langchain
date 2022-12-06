"""Unit tests for agents."""

from typing import Any, List, Mapping, Optional

from langchain.agents import Tool, initialize_agent
from langchain.llms.base import LLM


class FakeListLLM(LLM):
    """Fake LLM for testing that outputs elements of a list."""

    def __init__(self, responses: List[str]):
        """Initialize with list of responses."""
        self.responses = responses
        self.i = -1

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Increment counter, and then return response in that index."""
        self.i += 1
        print(self.i)
        print(self.responses)
        return self.responses[self.i]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


def test_agent_bad_action() -> None:
    """Test react chain when bad action given."""
    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
        Tool("Lookup", lambda x: x, "Useful for looking up things in a table"),
    ]
    agent = initialize_agent(
        tools, fake_llm, agent="zero-shot-react-description", verbose=True
    )
    output = agent.run("when was langchain made")
    assert output == "curses foiled again"
