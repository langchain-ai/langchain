from __future__ import annotations

import ast
import re
from typing import Sequence, List, Union

from langchain.automaton.chat_agent import ChatAgent
from langchain.automaton.tool_utils import generate_tool_info
from langchain.automaton.typedefs import (
    MessageLike,
    AgentFinish,
    FunctionCall,
    FunctionResult,
)
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

TEMPLATE_ = SystemMessagePromptTemplate.from_template(
    """Respond to the human as helpfully and accurately as \
possible. You have access to the following tools:
{tools_description}

Use a blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $BLOB, as shown.

<action>
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
</action>

When invoking a tool do not provide any clarifying information.

The human will forward results of tool invocations as "Observations".

When you know the answer paraphrase the information in the observations properly and respond to the user. \
If you do not know the answer use more tools.

You can only take a single action at a time."""
)


def get_start_state(tools: Sequence[BaseTool]) -> List[BaseMessage]:
    """Generate a prompt for the agent."""
    tool_info = generate_tool_info(tools)
    msg = TEMPLATE_.format(**tool_info)
    return [msg]


def _decode(text: Union[BaseMessage, str]) -> MessageLike:
    """Decode the action."""
    pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)
    if not isinstance(text, BaseMessage):
        raise NotImplementedError()
    _text = text.content
    match = pattern.search(_text)
    if match:
        action_blob = match.group("action_blob")
        data = ast.literal_eval(action_blob)
        name = data["action"]
        if name == "Final Answer":  # Special cased "tool" for final answer
            return AgentFinish(result=data["action_input"])
        return FunctionCall(
            name=data["action"], named_arguments=data["action_input"] or {}
        )
    else:
        return AgentFinish(result=text)


def generate_prompt(current_messages: Sequence[MessageLike]) -> List[BaseMessage]:
    """Generate a prompt from a log of message like objects."""
    messages = []
    for message in current_messages:
        if isinstance(message, BaseMessage):
            messages.append(message)
        elif isinstance(message, FunctionResult):
            messages.append(
                HumanMessage(
                    content=f"Observation: {message.result}",
                )
            )
        else:
            pass
    return messages


# PUBLIC API


def create_xml_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
) -> ChatAgent:
    """XML based chat agent."""
    return ChatAgent(
        llm=llm,
        tools=tools,
        prompt_generator=generate_prompt,
        parser=_decode,
    )
