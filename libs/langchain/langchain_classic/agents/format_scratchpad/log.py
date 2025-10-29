from langchain_core.agents import AgentAction


def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process.

    Args:
        intermediate_steps: List of tuples of AgentAction and observation strings.
        observation_prefix: Prefix to append the observation with.
        llm_prefix: Prefix to append the llm call with.

    Returns:
        The scratchpad.
    """
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
