from langchain_core.agents import AgentAction

from langchain.agents.format_scratchpad.xml import format_xml


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


def test_xml_escaping_minimal() -> None:
    """Test that XML tags in tool names are escaped with minimal format."""
    # Arrange
    agent_action = AgentAction(
        tool="search<tool>nested</tool>", tool_input="query<input>test</input>", log=""
    )
    observation = "Found <observation>result</observation>"
    intermediate_steps = [(agent_action, observation)]

    # Act
    result = format_xml(intermediate_steps, escape_format="minimal")

    # Assert - XML tags should be replaced with custom delimiters
    expected_result = (
        "<tool>search[[tool]]nested[[/tool]]</tool>"
        "<tool_input>query<input>test</input></tool_input>"
        "<observation>Found [[observation]]result[[/observation]]</observation>"
    )
    assert result == expected_result


def test_no_escaping() -> None:
    """Test that escaping can be disabled."""
    # Arrange
    agent_action = AgentAction(tool="Tool1", tool_input="Input1", log="")
    observation = "Observation1"
    intermediate_steps = [(agent_action, observation)]

    # Act
    result = format_xml(intermediate_steps, escape_format=None)

    # Assert
    expected_result = (
        "<tool>Tool1</tool><tool_input>Input1</tool_input>"
        "<observation>Observation1</observation>"
    )
    assert result == expected_result
