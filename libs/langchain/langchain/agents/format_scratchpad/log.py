from typing import List, Tuple

from langchain.schema.agent import AgentAction


def format_log(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix="Observation: ",
    llm_prefix="Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
