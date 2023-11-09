import json
from typing import List, Sequence, Tuple

from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from langchain.schema.agent import AgentAction
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)


def _convert_agent_action_to_messages(
    agent_action: AgentAction, observation: str
) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, OpenAIToolAgentAction):
        return list(agent_action.message_log) + [
            _create_tool_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]


def _create_tool_message(
    agent_action: OpenAIToolAgentAction, observation: str
) -> ToolMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
    )


def format_to_openai_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages
