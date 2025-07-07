import xml.sax.saxutils

from langchain_core.agents import AgentAction


def format_xml(
    intermediate_steps: list[tuple[AgentAction, str]],
    *,
    escape_xml: bool = True,
) -> str:
    """Format the intermediate steps as XML.

    Escapes all special XML characters in the content, preventing injection of malicious
    or malformed XML.

    Args:
        intermediate_steps: The intermediate steps, each a tuple of
            (AgentAction, observation).
        escape_xml: If True, all XML special characters in the tool name,
            tool input, and observation will be escaped (e.g., ``<`` becomes ``&lt;``).

    Returns:
        A string of concatenated XML blocks representing the intermediate steps.
    """
    log = ""
    for action, observation in intermediate_steps:
        tool = str(action.tool)
        tool_input = str(action.tool_input)
        observation_str = str(observation)

        if escape_xml:
            entities = {"'": "&apos;", '"': "&quot;"}
            tool = xml.sax.saxutils.escape(tool, entities)
            tool_input = xml.sax.saxutils.escape(tool_input, entities)
            observation_str = xml.sax.saxutils.escape(observation_str, entities)

        log += (
            f"<tool>{tool}</tool><tool_input>{tool_input}"
            f"</tool_input><observation>{observation_str}</observation>"
        )
    return log
