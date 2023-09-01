"""Module contains code for a general chat agent."""
from __future__ import annotations

import ast
import re
from typing import Sequence, Union, List

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


class ActionEncoder:
    def __init__(self) -> None:
        """Initialize the ActionParser."""
        self.pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)

    def decode(self, text: Union[BaseMessage, str]) -> MessageLike:
        """Decode the action."""
        if not isinstance(text, BaseMessage):
            raise NotImplementedError()
        _text = text.content
        match = self.pattern.search(_text)
        if match:
            action_blob = match.group("action_blob")
            data = ast.literal_eval(action_blob)
            name = data["action"]
            if name == "Final Answer":  # Special cased "tool" for final answer
                return AgentFinish(result=data["action_input"])
            return FunctionCall(
                name=data["action"], arguments=data["action_input"] or {}
            )
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
        action_encoder = ActionEncoder()
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=action_encoder.decode,
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
