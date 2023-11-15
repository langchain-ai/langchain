from langchain.agents.format_scratchpad.log_to_messages import format_log_to_messages
from langchain.schema.agent import AgentAction
from langchain.schema.messages import AIMessage, HumanMessage


def test_single_intermediate_step_default_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    expected_result = [AIMessage(content="Log1"), HumanMessage(content="Observation1")]
    assert format_log_to_messages(intermediate_steps) == expected_result


def test_multiple_intermediate_steps_default_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
        (AgentAction(tool="Tool2", tool_input="input2", log="Log2"), "Observation2"),
        (AgentAction(tool="Tool3", tool_input="input3", log="Log3"), "Observation3"),
    ]
    expected_result = [
        AIMessage(content="Log1"),
        HumanMessage(content="Observation1"),
        AIMessage(content="Log2"),
        HumanMessage(content="Observation2"),
        AIMessage(content="Log3"),
        HumanMessage(content="Observation3"),
    ]
    assert format_log_to_messages(intermediate_steps) == expected_result


def test_custom_template_tool_response() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    template_tool_response = "Response: {observation}"
    expected_result = [
        AIMessage(content="Log1"),
        HumanMessage(content="Response: Observation1"),
    ]
    assert (
        format_log_to_messages(
            intermediate_steps, template_tool_response=template_tool_response
        )
        == expected_result
    )


def test_empty_steps() -> None:
    assert format_log_to_messages([]) == []
