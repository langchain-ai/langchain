"""Unittests for langchain.agents.chat package."""

from langchain_core.agents import AgentAction

from langchain_classic.agents.chat.output_parser import ChatOutputParser

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
