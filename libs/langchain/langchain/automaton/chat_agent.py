"""Generalized chat agent, works with any chat model."""
from __future__ import annotations

import ast
import re
from typing import Sequence, Union, List
from langchain.automaton.tool_utils import generate_tool_info

from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    MessageLog,
    AgentFinish,
    MessageLike,
    FunctionCall,
    FunctionResult,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, HumanMessage
from langchain.tools import BaseTool

from langchain.prompts import SystemMessagePromptTemplate


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


def generate_prompt(tools: Sequence[BaseTool]) -> MessageLog:
    """Generate a prompt for the agent."""
    tool_info = generate_tool_info(tools)
    msg = TEMPLATE_.format(**tool_info)
    return MessageLog(messages=[msg])


def decode(text: Union[BaseMessage, str]) -> MessageLike:
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
        return FunctionCall(name=data["action"], named_arguments=data["action_input"] or {})
    else:
        return AgentFinish(result=text)


def prompt_generator(log: MessageLog) -> List[BaseMessage]:
    """Generate a prompt from a log of message like objects."""
    messages = []
    for message in log.messages:
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


class ChatAgent:
    """An agent for chat models."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        *,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the chat automaton."""
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=decode,
        )
        self.max_iterations = max_iterations

    def run(self, message_log: MessageLog) -> None:
        """Run the agent."""
        if not message_log:
            raise AssertionError(f"Expected at least one message in message_log")

        for _ in range(self.max_iterations):
            last_message = message_log[-1]

            if isinstance(last_message, AgentFinish):
                break

            messages = self.llm_program.invoke(message_log)
            message_log.add_messages(messages)
