"""Test MRKL functionality."""

import pytest

from langchain.chains.mrkl.base import get_action_and_input


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
