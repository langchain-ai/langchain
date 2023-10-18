from typing import List, Tuple

from langchain.schema.agent import AgentAction


def format_xml(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> str:
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log
