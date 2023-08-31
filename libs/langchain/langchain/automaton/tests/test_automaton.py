from __future__ import annotations

from typing import Sequence, List

from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
# from langchain.automaton.chat_automaton import ChatAutomaton
from langchain.automaton.mrkl_agent import ActionParser
from langchain.automaton.tests.utils import (
    FakeChatOpenAI,
    construct_func_invocation_message,
)
from langchain.schema import PromptValue
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    FunctionMessage,
)
from langchain.schema.runnable import RunnableLambda
from langchain.tools import tool, Tool
from langchain.tools.base import tool as tool_maker


def get_tools() -> List[Tool]:
    @tool
    def name() -> str:
        """Use to look up the user's name"""
        return "Eugene"

    @tool
    def get_weather(city: str) -> str:
        """Get weather in a specific city."""
        return "42F and sunny"

    @tool
    def add(x: int, y: int) -> int:
        """Use to add two numbers."""
        return x + y

    return list(locals().values())


def test_structured_output_chat() -> None:
    parser = StructuredChatOutputParser()
    output = parser.parse(
        """
        ```json
        {
            "action": "hello",
            "action_input": {
                "a": 2
            }
        }
        ```
        """
    )
    assert output == {}


class MessageBasedPromptValue(PromptValue):
    """Prompt Value populated from messages."""

    messages: List[BaseMessage]

    @classmethod
    def from_messages(cls, messages: Sequence[BaseMessage]) -> MessageBasedPromptValue:
        return cls(messages=messages)

    def to_messages(self) -> List[BaseMessage]:
        return self.messages

    def to_string(self) -> str:
        return "\n".join([message.content for message in self.messages])



# def prompt_generator(memory: Memory) -> PromptValue:
#     """Generate a prompt."""
#     if not memory.messages:
#         raise AssertionError("Memory is empty")
#     return MessageBasedPromptValue.from_messages(messages=memory.messages)
#

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


def test_generate_template() -> None:
    """Generate template."""
    template = generate_template()
    assert template.format_messages(tools="hello", tool_names="hello") == []


def test_parser() -> None:
    """Tes the parser."""
    sample_text = """
    Some text before
    <action>
    {
      "key": "value",
      "number": 42
    }
    </action>
    Some text after
    """
    action_parser = ActionParser(strict=False)
    action = action_parser.decode(sample_text)
    assert action == {
        "key": "value",
        "number": 42,
    }

def test_function_invocation() -> None:
    """test function invocation"""
    tools = get_tools()
    from langchain.automaton.well_known_states import create_tool_invoker
    runnable = create_tool_invoker(tools)
    result = runnable.invoke({"name": "add", "inputs": {"x": 1, "y": 2}})
    assert result == 3



def test_create_llm_program() -> None:
    """Generate llm program."""
    from langchain.automaton.mrkl_automaton import (
        _generate_prompt,
        _generate_mrkl_memory,
    )

    tools = get_tools()
    llm = FakeChatOpenAI(
        message_iter=iter(
            [
                AIMessage(
                    content="""Thought: Hello. <action>{"name": "key"}</action>""",
                ),
            ]
        )
    )

    program = create_llm_program(
        "think-act",
        llm,
        prompt_generator=_generate_prompt,
        stop=["Observation"],
        parser=RunnableLambda(ActionParser(strict=False).decode),
    )

    mrkl_memory = _generate_mrkl_memory(tools)
    result = program.invoke(mrkl_memory)
    assert result == {"id": "think-act", "data": {}}
