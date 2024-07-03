import json
from typing import List, Sequence, Tuple

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_glm.agent_toolkits import BaseToolOutput
from langchain_glm.agent_toolkits.all_tools.code_interpreter_tool import (
    CodeInterpreterToolOutput,
)
from langchain_glm.agent_toolkits.all_tools.drawing_tool import DrawingToolOutput
from langchain_glm.agent_toolkits.all_tools.web_browser_tool import (
    WebBrowserToolOutput,
)
from langchain_glm.agents.output_parsers.code_interpreter import (
    CodeInterpreterAgentAction,
)
from langchain_glm.agents.output_parsers.drawing_tool import DrawingToolAgentAction
from langchain_glm.agents.output_parsers.web_browser import WebBrowserAgentAction


def _create_tool_message(
    agent_action: ToolAgentAction, observation: BaseToolOutput
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
        additional_kwargs={"name": agent_action.tool},
    )


def format_to_zhipuai_all_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, BaseToolOutput]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, CodeInterpreterAgentAction):
            if isinstance(observation, CodeInterpreterToolOutput):
                if "auto" == observation.platform_params.get("sandbox", "auto"):
                    messages.append(AIMessage(content=str(observation)))
                elif "none" == observation.platform_params.get("sandbox", "auto"):
                    messages.append(_create_tool_message(agent_action, observation))
                else:
                    raise ValueError(
                        f"Unknown sandbox type: {observation.platform_params.get('sandbox', 'auto')}"
                    )
            else:
                raise ValueError(f"Unknown observation type: {type(observation)}")

        elif isinstance(agent_action, DrawingToolAgentAction):
            if isinstance(observation, DrawingToolOutput):
                new_messages = list(agent_action.message_log) + [
                    _create_tool_message(agent_action, observation)
                ]
                messages.extend([new for new in new_messages if new not in messages])
            else:
                raise ValueError(f"Unknown observation type: {type(observation)}")

        elif isinstance(agent_action, WebBrowserAgentAction):
            if isinstance(observation, WebBrowserToolOutput):
                new_messages = list(agent_action.message_log) + [
                    _create_tool_message(agent_action, observation)
                ]
                messages.extend([new for new in new_messages if new not in messages])
            else:
                raise ValueError(f"Unknown observation type: {type(observation)}")

        elif isinstance(agent_action, ToolAgentAction):
            new_messages = list(agent_action.message_log) + [
                _create_tool_message(agent_action, observation)
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages
