from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from typing_extensions import override

from langchain_classic.agents.agent import MultiActionAgentOutputParser
from langchain_classic.agents.output_parsers.tools import (
    ToolAgentAction,
    parse_ai_message_to_tool_action,
)

OpenAIToolAgentAction = ToolAgentAction


def parse_ai_message_to_openai_tool_action(
    message: BaseMessage,
) -> list[AgentAction] | AgentFinish:
    """Parse an AI message potentially containing tool_calls."""
    tool_actions = parse_ai_message_to_tool_action(message)
    if isinstance(tool_actions, AgentFinish):
        return tool_actions
    final_actions: list[AgentAction] = []
    for action in tool_actions:
        if isinstance(action, ToolAgentAction):
            final_actions.append(
                OpenAIToolAgentAction(
                    tool=action.tool,
                    tool_input=action.tool_input,
                    log=action.log,
                    message_log=action.message_log,
                    tool_call_id=action.tool_call_id,
                ),
            )
        else:
            final_actions.append(action)
    return final_actions


class OpenAIToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    tool_calls parameter from OpenAI to convey what tools to use.

    If a tool_calls parameter is passed, then that is used to get
    the tool names and tool inputs.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "openai-tools-agent-output-parser"

    @override
    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False,
    ) -> list[AgentAction] | AgentFinish:
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)  # noqa: TRY004
        message = result[0].message
        return parse_ai_message_to_openai_tool_action(message)

    @override
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        msg = "Can only parse messages"
        raise ValueError(msg)
