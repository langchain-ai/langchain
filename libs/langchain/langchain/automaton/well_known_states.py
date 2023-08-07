from __future__ import annotations

import dataclasses
from typing import Sequence

from langchain.automaton.automaton import State, ExecutedState
from langchain.automaton.open_ai_functions import create_action_taking_llm
from langchain.schema import HumanMessage, BaseMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


@dataclasses.dataclass
class FunctionInvocation(State):
    llm: BaseLanguageModel
    tools: Sequence[BaseTool]


@dataclasses.dataclass
class LLMProgram(State):
    """A state that executes an LLM program."""

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    # # This should either be swapped with memory or else with prompt value?
    # # Likely prompt value since we're not taking in any input
    messages: Sequence[BaseMessage]  # Swap with prompt value
    # memory: Memory
    # prompt_generator: PromptGenerator

    def execute(self) -> ExecutedState:
        """Execute LLM program."""
        action_taking_llm = create_action_taking_llm(self.llm, tools=self.tools)
        messages = self.messages
        # prompt_value = self.prompt_generator(self.memory)
        # result = action_taking_llm.invoke(prompt_value)
        # self.memory.add_message(result["message"])
        result = action_taking_llm.invoke(messages)
        return {"id": "llm_program", "data": result}


@dataclasses.dataclass
class UserInputState(State):
    """A state that prompts the user for input from stdin.

    This is primarily useful for interactive development.
    """

    def execute(self) -> ExecutedState:
        """Execute user input state."""
        user_input = input("Enter your input: ")
        return {
            "id": "user_input",
            "data": {
                "message": HumanMessage(content=user_input),
            },
        }
