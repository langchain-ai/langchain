"""Test MRKL functionality."""

import pytest

from langchain.chains.mrkl.base import ChainConfig, MRKLChain, get_action_and_input
from langchain.chains.mrkl.prompt import BASE_TEMPLATE
from langchain.prompts import Prompt
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_get_action_and_input() -> None:
    """Test getting an action from text."""
    llm_output = (
        "Thought: I need to search for NBA\n" "Action: Search\n" "Action Input: NBA"
    )
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
    assert action == "Final Answer: "
    assert action_input == "1994"


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
        ChainConfig(
            action_name="foo", action=lambda x: "foo", action_description="foobar1"
        ),
        ChainConfig(
            action_name="bar", action=lambda x: "bar", action_description="foobar2"
        ),
    ]
    mrkl_chain = MRKLChain.from_chains(FakeLLM(), chain_configs)
    expected_tools_prompt = "foo: foobar1\nbar: foobar2"
    expected_tool_names = "foo, bar"
    expected_template = BASE_TEMPLATE.format(
        tools=expected_tools_prompt, tool_names=expected_tool_names
    )
    prompt = mrkl_chain.prompt
    assert isinstance(prompt, Prompt)
    assert prompt.template == expected_template
