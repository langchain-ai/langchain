"""Test MRKL functionality."""

from typing import Tuple

import pytest

from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, OutputParserException
from tests.unit_tests.llms.fake_llm import FakeLLM


def get_action_and_input(text: str) -> Tuple[str, str]:
    output = MRKLOutputParser().parse(text)
    if isinstance(output, AgentAction):
        return output.tool, str(output.tool_input)
    else:
        return "Final Answer", output.return_values["output"]


def test_get_action_and_input() -> None:
    """Test getting an action from text."""
    llm_output = (
        "Thought: I need to search for NBA\n" "Action: Search\n" "Action Input: NBA"
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Search"
    assert action_input == "NBA"


def test_get_action_and_input_whitespace() -> None:
    """Test getting an action from text."""
    llm_output = "Thought: I need to search for NBA\nAction: Search \nAction Input: NBA"
    action, action_input = get_action_and_input(llm_output)
    assert action == "Search"
    assert action_input == "NBA"


def test_get_action_and_input_newline() -> None:
    """Test getting an action from text where Action Input is a code snippet."""
    llm_output = (
        "Now I need to write a unittest for the function.\n\n"
        "Action: Python\nAction Input:\n```\nimport unittest\n\nunittest.main()\n```"
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Python"
    assert action_input == "```\nimport unittest\n\nunittest.main()\n```"


def test_get_action_and_input_newline_after_keyword() -> None:
    """Test getting an action and action input from the text
    when there is a new line before the action
    (after the keywords "Action:" and "Action Input:")
    """
    llm_output = """
    I can use the `ls` command to list the contents of the directory \
    and `grep` to search for the specific file.

    Action:
    Terminal

    Action Input:
    ls -l ~/.bashrc.d/
    """

    action, action_input = get_action_and_input(llm_output)
    assert action == "Terminal"
    assert action_input == "ls -l ~/.bashrc.d/\n"


def test_get_action_and_input_sql_query() -> None:
    """Test getting the action and action input from the text
    when the LLM output is a well formed SQL query
    """
    llm_output = """
    I should query for the largest single shift payment for every unique user.
    Action: query_sql_db
    Action Input: \
    SELECT "UserName", MAX(totalpayment) FROM user_shifts GROUP BY "UserName" """
    action, action_input = get_action_and_input(llm_output)
    assert action == "query_sql_db"
    assert (
        action_input
        == 'SELECT "UserName", MAX(totalpayment) FROM user_shifts GROUP BY "UserName"'
    )


def test_get_final_answer() -> None:
    """Test getting final answer."""
    llm_output = "Thought: I can now answer the question\n" "Final Answer: 1994"
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_new_line() -> None:
    """Test getting final answer."""
    llm_output = "Thought: I can now answer the question\n" "Final Answer:\n1994"
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_multiline() -> None:
    """Test getting final answer that is multiline."""
    llm_output = "Thought: I can now answer the question\n" "Final Answer: 1994\n1993"
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994\n1993"


def test_bad_action_input_line() -> None:
    """Test handling when no action input found."""
    llm_output = "Thought: I need to search for NBA\n" "Action: Search\n" "Thought: NBA"
    with pytest.raises(OutputParserException) as e_info:
        get_action_and_input(llm_output)
    assert e_info.value.observation is not None


def test_bad_action_line() -> None:
    """Test handling when no action found."""
    llm_output = (
        "Thought: I need to search for NBA\n" "Thought: Search\n" "Action Input: NBA"
    )
    with pytest.raises(OutputParserException) as e_info:
        get_action_and_input(llm_output)
    assert e_info.value.observation is not None


def test_valid_action_and_answer_raises_exception() -> None:
    """Test handling when both an action and answer are found."""
    llm_output = (
        "Thought: I need to search for NBA\n"
        "Action: Search\n"
        "Action Input: NBA\n"
        "Observation: founded in 1994\n"
        "Thought: I can now answer the question\n"
        "Final Answer: 1994"
    )
    with pytest.raises(OutputParserException):
        get_action_and_input(llm_output)


def test_from_chains() -> None:
    """Test initializing from chains."""
    chain_configs = [
        Tool(name="foo", func=lambda x: "foo", description="foobar1"),
        Tool(name="bar", func=lambda x: "bar", description="foobar2"),
    ]
    agent = ZeroShotAgent.from_llm_and_tools(FakeLLM(), chain_configs)
    expected_tools_prompt = "foo: foobar1\nbar: foobar2"
    expected_tool_names = "foo, bar"
    expected_template = "\n\n".join(
        [
            PREFIX,
            expected_tools_prompt,
            FORMAT_INSTRUCTIONS.format(tool_names=expected_tool_names),
            SUFFIX,
        ]
    )
    prompt = agent.llm_chain.prompt
    assert isinstance(prompt, PromptTemplate)
    assert prompt.template == expected_template
