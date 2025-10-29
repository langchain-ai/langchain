from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_log_to_messages(
    intermediate_steps: list[tuple[AgentAction, str]],
    template_tool_response: str = "{observation}",
) -> list[BaseMessage]:
    """Construct the scratchpad that lets the agent continue its thought process.

    Args:
        intermediate_steps: List of tuples of AgentAction and observation strings.
        template_tool_response: Template to format the observation with.
            Defaults to `"{observation}"`.

    Returns:
        The scratchpad.
    """
    thoughts: list[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(observation=observation),
        )
        thoughts.append(human_message)
    return thoughts
