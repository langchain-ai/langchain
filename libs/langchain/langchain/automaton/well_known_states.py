from __future__ import annotations

import dataclasses
from typing import Sequence

from langchain.automaton.automaton import State, ExecutedState
from langchain.automaton.open_ai_functions import create_action_taking_llm
from langchain.automaton.typedefs import (
    Memory,
    PromptGenerator,
    infer_message_type,
    MessageType,
)
from langchain.schema import HumanMessage, FunctionMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


@dataclasses.dataclass
class LLMProgram(State):
    """A state that executes an LLM program."""

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt_generator: PromptGenerator

    def execute(self, memory: Memory) -> ExecutedState:
        """Execute LLM program."""
        action_taking_llm = create_action_taking_llm(self.llm, tools=self.tools)
        prompt_value = self.prompt_generator(memory)
        result = action_taking_llm.invoke(prompt_value)
        # Memory is mutable
        message = result["message"]
        if not isinstance(message, AIMessage):
            raise AssertionError(
                f"LLM program should return an AI message. Got a {type(message)}."
            )
        memory.add_message(message)

        if infer_message_type(message) == MessageType.AI_INVOKE:
            data = result["data"]
            function_message = FunctionMessage(
                name=data["function_call"]["name"],
                content=data["function_call"]["result"],
            )
            memory.add_message(function_message)

        # What information should the state return in this case.
        # Does it matter, folks can use it or not...
        return {"id": "llm_program", "data": result}


@dataclasses.dataclass
class UserInputState(State):
    """A state that prompts the user for input from stdin.

    This is primarily useful for interactive development.
    """

    def execute(self, memory: Memory) -> ExecutedState:
        """Execute user input state."""
        user_input = input("Enter your input: ")
        message = HumanMessage(content=user_input)
        memory.add_message(message)

        return {
            "id": "user_input",
            "data": {
                "message": message,
            },
        }
