"""Unittests for langchain.agents.chat package."""
from typing import Tuple

from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.schema import AgentAction, AgentFinish

output_parser = StructuredChatOutputParser()


def get_action_and_input(text: str) -> Tuple[str, str]:
    output = output_parser.parse(text)
    if isinstance(output, AgentAction):
        return output.tool, str(output.tool_input)
    elif isinstance(output, AgentFinish):
        return output.return_values["output"], output.log
    else:
        raise ValueError("Unexpected output type")


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


def test_parse_with_language_and_spaces() -> None:
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


def test_parse_without_language_without_a_new_line() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.

    Action:
    ```{"action": "foo", "action_input": "bar"}```
    """
    action, action_input = get_action_and_input(llm_output)
    assert action == "foo"
    assert action_input == "bar"


def test_parse_with_language_without_a_new_line() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.

    Action:
    ```json{"action": "foo", "action_input": "bar"}```
    """
    # TODO: How should this be handled?
    output, log = get_action_and_input(llm_output)
    assert output == llm_output
    assert log == llm_output


def test_parse_case_matched_and_final_answer() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.

    Action:
    ```json
    {
      "action": "Final Answer",
      "action_input": "This is the final answer"
    }
    ```
    """
    output, log = get_action_and_input(llm_output)
    assert output == "This is the final answer"
    assert log == llm_output
