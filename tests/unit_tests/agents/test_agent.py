"""Unit tests for agents."""

from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from langchain.agents import Tool, initialize_agent
from langchain.llms.base import LLM


class FakeListLLM(LLM, BaseModel):
    """Fake LLM for testing that outputs elements of a list."""

    responses: List[str]
    i: int = -1

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Increment counter, and then return response in that index."""
        self.i += 1
        print(self.i)
        print(self.responses)
        return self.responses[self.i]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake_list"


def test_agent_bad_action() -> None:
    """Test react chain when bad action given."""
    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
        Tool("Lookup", lambda x: x, "Useful for looking up things in a table"),
    ]
    agent = initialize_agent(
        tools, fake_llm, agent="zero-shot-react-description", verbose=True
    )
    output = agent.run("when was langchain made")
    assert output == "curses foiled again"
