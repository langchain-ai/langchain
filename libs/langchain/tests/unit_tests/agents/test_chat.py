"""Unittests for langchain.agents.chat package."""

from pathlib import Path

import pytest
from langchain_core.agents import AgentAction
from langchain_core.tools import Tool

from langchain_classic.agents.chat.base import ChatAgent
from langchain_classic.agents.chat.output_parser import ChatOutputParser
from tests.unit_tests.llms.fake_llm import FakeLLM

output_parser = ChatOutputParser()


def get_action_and_input(text: str) -> tuple[str, str]:
    output = output_parser.parse(text)
    if isinstance(output, AgentAction):
        return output.tool, str(output.tool_input)
    return "Final Answer", output.return_values["output"]


def test_parse_with_language() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.

    Action:
    ```json
    {
      "action": "foo",
      "action_input": "bar"
    }
    ```
    """
    action, action_input = get_action_and_input(llm_output)
    assert action == "foo"
    assert action_input == "bar"


def test_parse_without_language() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.

    Action:
    ```
    {
      "action": "foo",
      "action_input": "bar"
    }
    ```
    """
    action, action_input = get_action_and_input(llm_output)
    assert action == "foo"
    assert action_input == "bar"


def test_chat_agent_save_reports_unsupported(tmp_path: Path) -> None:
    agent = ChatAgent.from_llm_and_tools(
        FakeLLM(),
        [Tool(name="foo", description="Test tool FOO", func=lambda x: x)],
    )

    with pytest.raises(NotImplementedError, match="does not support saving"):
        agent.save(tmp_path / "agent.yaml")
