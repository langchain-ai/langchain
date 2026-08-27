import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.agents import AgentAction
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_classic.agents.output_parsers.tools import ToolAgentAction

_logger = logging.getLogger(__name__)


def _create_tool_message(
    agent_action: ToolAgentAction,
    observation: Any,
) -> ToolMessage:
    """Convert agent action and observation into a tool message.

    Args:
        agent_action: the tool invocation request from the agent.
        observation: the result of the tool invocation.

    Returns:
        ToolMessage that corresponds to the original tool invocation.

    Raises:
        ValueError: if the observation cannot be converted to a string.
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except TypeError:
            content = str(observation)
        except Exception:
            _logger.exception("Unexpected error converting observation to string.")
            content = str(observation)
    else:
        content = observation
    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs={"name": agent_action.tool},
    )


def format_to_tool_messages(
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into `ToolMessage` objects.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations.

    Returns:
        list of messages to send to the LLM for the next prediction.

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = [
                *list(agent_action.message_log),
                _create_tool_message(agent_action, observation),
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages
