from langchain_core.agents import AgentAction


def format_xml(
    intermediate_steps: list[tuple[AgentAction, str]],
) -> str:
    """Format the intermediate steps as XML.

    Args:
        intermediate_steps: The intermediate steps.

    Returns:
        The intermediate steps as XML.
    """
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log
