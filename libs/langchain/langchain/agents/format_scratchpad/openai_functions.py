import json
from collections.abc import Sequence

from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
    observation: str,
) -> list[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage or the previous messages plus a FunctionMessage that corresponds to
            the original tool invocation
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [
            *list(agent_action.message_log),
            _create_function_message(agent_action, observation),
        ]
    return [AIMessage(content=agent_action.log)]


def _create_function_message(
    agent_action: AgentAction,
    observation: str,
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent.
        observation: the result of the tool invocation.
    Returns:
        FunctionMessage that corresponds to the original tool invocation.

    Raises:
        ValueError: if the observation cannot be converted to a string.
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def format_to_openai_function_messages(
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction
    Raises:
        ValueError: if the observation cannot be converted to a string.
    """
    messages = []

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages


# Backwards compatibility
format_to_openai_functions = format_to_openai_function_messages
