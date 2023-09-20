from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.schema.agent import AgentAction


def test_single_agent_action_observation() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    expected_result = "Log1\nObservation: Observation1\nThought: "
    assert format_log_to_str(intermediate_steps) == expected_result


def test_multiple_agent_actions_observations() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
        (AgentAction(tool="Tool2", tool_input="input2", log="Log2"), "Observation2"),
        (AgentAction(tool="Tool3", tool_input="input3", log="Log3"), "Observation3"),
    ]
    expected_result = """Log1\nObservation: Observation1\nThought: \
Log2\nObservation: Observation2\nThought: Log3\nObservation: \
Observation3\nThought: """
    assert format_log_to_str(intermediate_steps) == expected_result


def test_custom_prefixes() -> None:
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    observation_prefix = "Custom Observation: "
    llm_prefix = "Custom Thought: "
    expected_result = "Log1\nCustom Observation: Observation1\nCustom Thought: "
    assert (
        format_log_to_str(intermediate_steps, observation_prefix, llm_prefix)
        == expected_result
    )


def test_empty_intermediate_steps() -> None:
    output = format_log_to_str([])
    assert output == ""
