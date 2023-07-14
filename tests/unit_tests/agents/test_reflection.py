"""Unit tests for reflection."""

from langchain.agents.reflexion.react import ReactReflector
from langchain.llms.fake import FakeListLLM
from langchain.schema import AgentAction


def test_should_continue() -> None:
    """Test react chain with callbacks by setting verbose globally."""

    llm = FakeListLLM(responses=["test response"])
    reflector = ReactReflector.from_llm(
        llm=llm,
        max_iterations_per_trial=5,
        max_execution_time_per_trial=10.0,
        max_action_repetition=2,
    )

    non_repeating_actions = [
        (
            AgentAction(tool="FirstTestTool", tool_input="test input", log="I did it!"),
            "First result",
        ),
        (
            AgentAction(
                tool="SecondTestTool", tool_input="test input", log="I did it!"
            ),
            "Second result",
        ),
    ]
    repeating_actions = [
        (
            AgentAction(tool="FirstTestTool", tool_input="test input", log="I did it!"),
            "First result",
        ),
        (
            AgentAction(
                tool="SecondTestTool", tool_input="test input", log="I did it!"
            ),
            "Second result",
        ),
        (
            AgentAction(
                tool="SecondTestTool", tool_input="test input", log="I did it!"
            ),
            "Third result",
        ),
    ]

    # trial failed due to max iterations exceeded
    assert (
        reflector.should_reflect(
            iterations_in_trial=20,
            execution_time_in_trial=5.0,
            intermediate_steps=non_repeating_actions,
        )
        is True
    )

    # trial failed due to max execution time exceeded
    assert (
        reflector.should_reflect(
            iterations_in_trial=3,
            execution_time_in_trial=20.0,
            intermediate_steps=non_repeating_actions,
        )
        is True
    )

    # trial failed due to max action repetitions exceeded
    assert (
        reflector.should_reflect(
            iterations_in_trial=3,
            execution_time_in_trial=5.0,
            intermediate_steps=repeating_actions,
        )
        is True
    )

    # trial didn't fail
    assert (
        reflector.should_reflect(
            iterations_in_trial=3,
            execution_time_in_trial=5.0,
            intermediate_steps=non_repeating_actions,
        )
        is False
    )


def test_reflection() -> None:
    test_response = "test response"

    llm = FakeListLLM(responses=[test_response])
    reflector = ReactReflector.from_llm(
        llm=llm,
        max_iterations_per_trial=5,
        max_execution_time_per_trial=10.0,
        max_action_repetition=2,
    )

    assert (
        reflector.reflect(
            input="Test question",
            current_trial="String representation of current trial",
            current_trial_no=1,
        )
        == test_response
    )
