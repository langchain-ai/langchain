"""Unittests for langchain.agents.chat package."""

from textwrap import dedent
from typing import Any

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import Tool

from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser

output_parser = StructuredChatOutputParser()


def get_action_and_input(text: str) -> tuple[str, str]:
    output = output_parser.parse(text)
    if isinstance(output, AgentAction):
        return output.tool, str(output.tool_input)
    if isinstance(output, AgentFinish):
        return output.return_values["output"], output.log
    msg = "Unexpected output type"  # type: ignore[unreachable]
    raise ValueError(msg)


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


# TODO: add more tests.
# Test: StructuredChatAgent.create_prompt() method.
class TestCreatePrompt:
    # Test: Output should be a ChatPromptTemplate with sys and human messages.
    def test_create_prompt_output(self) -> None:
        prompt = StructuredChatAgent.create_prompt(
            [Tool(name="foo", description="Test tool FOO", func=lambda x: x)],
        )

        assert isinstance(prompt, ChatPromptTemplate)
        assert len(prompt.messages) == 2
        assert isinstance(prompt.messages[0], SystemMessagePromptTemplate)
        assert isinstance(prompt.messages[1], HumanMessagePromptTemplate)

    # Test: Format with a single tool.
    def test_system_message_single_tool(self) -> None:
        prompt: Any = StructuredChatAgent.create_prompt(
            [Tool(name="foo", description="Test tool FOO", func=lambda x: x)],
        )
        actual = prompt.messages[0].prompt.format()

        expected = dedent(
            """
            Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            foo: Test tool FOO, args: {'tool_input': {'type': 'string'}}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or foo

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {
              "action": "Final Answer",
              "action_input": "Final response to human"
            }
            ```

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
            Thought:
            """,  # noqa: E501
        ).strip()

        assert actual == expected

    # Test: Format with multiple tools.
    #
    # Check:
    #
    #   You have access to the following tools:
    #   ...
    #
    # and
    #
    #   Valid "action" values: "Final Answer" or ...
    #
    def test_system_message_multiple_tools(self) -> None:
        prompt: Any = StructuredChatAgent.create_prompt(
            [
                Tool(name="foo", description="Test tool FOO", func=lambda x: x),
                Tool(name="bar", description="Test tool BAR", func=lambda x: x),
            ],
        )

        actual = prompt.messages[0].prompt.format()

        expected = dedent(
            """
            Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            foo: Test tool FOO, args: {'tool_input': {'type': 'string'}}
            bar: Test tool BAR, args: {'tool_input': {'type': 'string'}}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or foo, bar

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {
              "action": "Final Answer",
              "action_input": "Final response to human"
            }
            ```

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
            Thought:
            """,  # noqa: E501
        ).strip()

        assert actual == expected
