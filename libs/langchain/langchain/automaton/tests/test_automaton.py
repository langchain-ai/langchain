from __future__ import annotations

from typing import (
    List,
    Sequence,
)

from langchain.automaton.chat_automaton import ChatAutomaton
from langchain.automaton.executor import Executor
from langchain.automaton.tests.utils import (
    FakeChatOpenAI,
    construct_func_invocation_message,
)
from langchain.automaton.typedefs import Memory
from langchain.schema import PromptValue
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    FunctionMessage,
)
from langchain.tools.base import tool as tool_maker


class MessageBasedPromptValue(PromptValue):
    """Prompt Value populated from messages."""

    messages: List[BaseMessage]

    @classmethod
    def from_messages(cls, messages: Sequence[BaseMessage]) -> MessageBasedPromptValue:
        return cls(messages=messages)

    def to_messages(self) -> List[BaseMessage]:
        return self.messages

    def to_string(self) -> str:
        return " ".join([message.content for message in self.messages])


def prompt_generator(memory: Memory) -> PromptValue:
    """Generate a prompt."""
    if not memory.messages:
        raise AssertionError("Memory is empty")
    return MessageBasedPromptValue.from_messages(messages=memory.messages)


def test_automaton() -> None:
    """Run the automaton."""

    @tool_maker
    def get_time() -> str:
        """Get time."""
        return "9 PM"

    @tool_maker
    def get_location() -> str:
        """Get location."""
        return "the park"

    tools = [get_time, get_location]
    llm = FakeChatOpenAI(
        message_iter=iter(
            [
                construct_func_invocation_message(get_time, {}),
                AIMessage(
                    content="The time is 9 PM.",
                ),
            ]
        )
    )

    # TODO(FIX MUTABILITY)

    memory = Memory(
        messages=[
            SystemMessage(
                content=(
                    "Hello! I'm a chatbot that can help you write a letter. "
                    "What would you like to do?"
                ),
            )
        ]
    )
    chat_automaton = ChatAutomaton(
        llm=llm, tools=tools, prompt_generator=prompt_generator
    )
    executor = Executor(chat_automaton, memory, max_iterations=1)
    state, executed_states = executor.run()
    assert executed_states == [
        {
            "data": {
                "message": FunctionMessage(
                    content="9 PM", additional_kwargs={}, name="get_time"
                )
            },
            "id": "llm_program",
        }
    ]
