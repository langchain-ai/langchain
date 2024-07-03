from typing import List, Union

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_glm.agents.output_parsers.code_interpreter import (
    CodeInterpreterAgentAction,
)
from langchain_glm.agents.output_parsers.drawing_tool import DrawingToolAgentAction
from langchain_glm.agents.output_parsers.tools import (
    parse_ai_message_to_tool_action,
)
from langchain_glm.agents.output_parsers.web_browser import WebBrowserAgentAction

ZhipuAiALLToolAgentAction = ToolAgentAction


def parse_ai_message_to_zhipuai_all_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    tool_actions = parse_ai_message_to_tool_action(message)
    if isinstance(tool_actions, AgentFinish):
        return tool_actions
    final_actions: List[AgentAction] = []
    for action in tool_actions:
        if isinstance(action, CodeInterpreterAgentAction):
            final_actions.append(action)
        elif isinstance(action, DrawingToolAgentAction):
            final_actions.append(action)
        elif isinstance(action, WebBrowserAgentAction):
            final_actions.append(action)
        elif isinstance(action, ToolAgentAction):
            final_actions.append(
                ZhipuAiALLToolAgentAction(
                    tool=action.tool,
                    tool_input=action.tool_input,
                    log=action.log,
                    message_log=action.message_log,
                    tool_call_id=action.tool_call_id,
                )
            )
        else:
            final_actions.append(action)
    return final_actions


class ZhipuAiALLToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    tool_calls parameter from OpenAI to convey what tools to use.

    If a tool_calls parameter is passed, then that is used to get
    the tool names and tool inputs.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "zhipuai-all-tools-agent-output-parser"

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return parse_ai_message_to_zhipuai_all_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
