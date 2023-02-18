"""Test MRKL functionality."""

import pytest

from langchain.agents.mrkl.base import ZeroShotAgent, get_action_and_input
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate
from tests.unit_tests.llms.fake_llm import FakeLLM


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


def test_get_final_answer() -> None:
    """Test getting final answer."""
    llm_output = (
        "Thought: I need to search for NBA\n"
        "Action: Search\n"
        "Action Input: NBA\n"
        "Observation: founded in 1994\n"
        "Thought: I can now answer the question\n"
        "Final Answer: 1994"
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_new_line() -> None:
    """Test getting final answer."""
    llm_output = (
        "Thought: I need to search for NBA\n"
        "Action: Search\n"
        "Action Input: NBA\n"
        "Observation: founded in 1994\n"
        "Thought: I can now answer the question\n"
        "Final Answer:\n1994"
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_multiline() -> None:
    """Test getting final answer that is multiline."""
    llm_output = (
        "Thought: I need to search for NBA\n"
        "Action: Search\n"
        "Action Input: NBA\n"
        "Observation: founded in 1994 and 1993\n"
        "Thought: I can now answer the question\n"
        "Final Answer: 1994\n1993"
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994\n1993"


def test_bad_action_input_line() -> None:
    """Test handling when no action input found."""
    llm_output = "Thought: I need to search for NBA\n" "Action: Search\n" "Thought: NBA"
    with pytest.raises(ValueError):
        get_action_and_input(llm_output)


def test_bad_action_line() -> None:
    """Test handling when no action input found."""
    llm_output = (
        "Thought: I need to search for NBA\n" "Thought: Search\n" "Action Input: NBA"
    )
    with pytest.raises(ValueError):
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
