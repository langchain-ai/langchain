from langchain_core.agents import AgentAction

from langchain.agents.format_scratchpad.xml import format_xml


def test_format_single_action() -> None:
    """Tests formatting of a single agent action and observation."""
    agent_action = AgentAction(tool="Tool1", tool_input="Input1", log="Log1")
    intermediate_steps = [(agent_action, "Observation1")]

    result = format_xml(intermediate_steps, escape_xml=False)
    expected = (
        "<tool>Tool1</tool><tool_input>Input1</tool_input>"
        "<observation>Observation1</observation>"
    )
    assert result == expected


def test_format_xml_escaping() -> None:
    """Tests that XML special characters in content are properly escaped."""
    agent_action = AgentAction(
        tool="<tool> & 'some_tool'", tool_input='<query> with "quotes"', log=""
    )
    observation = "Observed > 5 items"
    intermediate_steps = [(agent_action, observation)]

    result = format_xml(intermediate_steps, escape_xml=True)

    expected = (
        "<tool>&lt;tool&gt; &amp; &apos;some_tool&apos;</tool>"
        "<tool_input>&lt;query&gt; with &quot;quotes&quot;</tool_input>"
        "<observation>Observed &gt; 5 items</observation>"
    )
    assert result == expected
