from langchain.agents.format_scratchpad.log import format_log
from langchain.schema.agent import AgentAction


def test_single_agent_action_observation():
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    expected_result = "Log1\nObservation: Observation1\nThought: "
    assert format_log(intermediate_steps) == expected_result


def test_multiple_agent_actions_observations():
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1"),
        (AgentAction(tool="Tool2", tool_input="input2", log="Log2"), "Observation2"),
        (AgentAction(tool="Tool3", tool_input="input3", log="Log3"), "Observation3"),
    ]
    expected_result = "Log1\nObservation: Observation1\nThought: \nLog2\nObservation: Observation2\nThought: \nLog3\nObservation: Observation3\nThought: "
    assert format_log(intermediate_steps) == expected_result


def test_custom_prefixes():
    intermediate_steps = [
        (AgentAction(tool="Tool1", tool_input="input1", log="Log1"), "Observation1")
    ]
    observation_prefix = "Custom Observation: "
    llm_prefix = "Custom Thought: "
    expected_result = "Log1\nCustom Observation: Observation1\nCustom Thought: "
    assert (
        format_log(intermediate_steps, observation_prefix, llm_prefix)
        == expected_result
    )


def test_empty_intermediate_steps():
    intermediate_steps = []
    expected_result = ""
    assert format_log(intermediate_steps) == expected_result
