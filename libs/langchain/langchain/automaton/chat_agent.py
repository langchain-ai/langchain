"""Module contains code for a general chat agent."""
from __future__ import annotations

import ast
import re
from typing import Sequence, Union, Optional

from langchain.automaton.prompt_generators import MessageLogPromptValue
from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    MessageLog,
    AgentFinish, MessageLike, FunctionCall,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage
from langchain.tools import BaseTool


class ActionParser:
    """A utility class to encode and decode action blocks."""

    def __init__(self) -> None:
        """Initialize the ActionParser."""
        self.pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)

    def decode(self, text: Union[BaseMessage, str]) -> Optional[MessageLike]:
        """Decode the action."""
        if isinstance(text, BaseMessage):
            text = text.content
        match = self.pattern.search(text)
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
            return None


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
            prompt_generator=MessageLogPromptValue.from_message_log,
            tools=tools,
            parser=OpenAIFunctionsParser(),
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
