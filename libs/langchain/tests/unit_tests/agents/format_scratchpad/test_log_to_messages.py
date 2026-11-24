from langchain_core.agents import AgentAction

from langchain_classic.agents.format_scratchpad.log_to_messages import (
    format_log_to_messages,
)
from tests.unit_tests.stubs import _AnyIdAIMessage, _AnyIdHumanMessage


def test_single_intermediate_step_default_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
    ]
    expected_result = [
        _AnyIdAIMessage(content="Log1"),
        _AnyIdHumanMessage(content="Observation1"),
    ]
    assert format_log_to_messages(intermediate_steps) == expected_result


def test_multiple_intermediate_steps_default_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
        (AgentAction(tool="Tool2", tool_input="input2", log="Log2"), "Observation2"),
        (AgentAction(tool="Tool3", tool_input="input3", log="Log3"), "Observation3"),
    ]
    expected_result = [
        _AnyIdAIMessage(content="Log1"),
        _AnyIdHumanMessage(content="Observation1"),
        _AnyIdAIMessage(content="Log2"),
        _AnyIdHumanMessage(content="Observation2"),
        _AnyIdAIMessage(content="Log3"),
        _AnyIdHumanMessage(content="Observation3"),
    ]
    assert format_log_to_messages(intermediate_steps) == expected_result


def test_custom_template_tool_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
    ]
    template_tool_response = "Response: {observation}"
    expected_result = [
        _AnyIdAIMessage(content="Log1"),
        _AnyIdHumanMessage(content="Response: Observation1"),
    ]
    assert (
        format_log_to_messages(
            intermediate_steps,
            template_tool_response=template_tool_response,
        )
        == expected_result
    )


def test_empty_steps() -> None:
    assert format_log_to_messages([]) == []
