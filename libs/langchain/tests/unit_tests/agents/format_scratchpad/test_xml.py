from langchain.agents.format_scratchpad.xml import format_xml
from langchain.schema.agent import AgentAction


def test_single_agent_action_observation() -> None:
    # Arrange
    agent_action = AgentAction(tool="Tool1", tool_input="Input1", log="Log1")
    observation = "Observation1"
    intermediate_steps = [(agent_action, observation)]

    # Act
    result = format_xml(intermediate_steps)
    expected_result = """<tool>Tool1</tool><tool_input>Input1\
</tool_input><observation>Observation1</observation>"""
    # Assert
    assert result == expected_result


def test_multiple_agent_actions_observations() -> None:
    # Arrange
    agent_action1 = AgentAction(tool="Tool1", tool_input="Input1", log="Log1")
    agent_action2 = AgentAction(tool="Tool2", tool_input="Input2", log="Log2")
    observation1 = "Observation1"
    observation2 = "Observation2"
    intermediate_steps = [(agent_action1, observation1), (agent_action2, observation2)]

    # Act
    result = format_xml(intermediate_steps)

    # Assert
    expected_result = """<tool>Tool1</tool><tool_input>Input1\
</tool_input><observation>Observation1</observation><tool>\
Tool2</tool><tool_input>Input2</tool_input><observation>\
Observation2</observation>"""
    assert result == expected_result


def test_empty_list_agent_actions() -> None:
    result = format_xml([])
    assert result == ""
