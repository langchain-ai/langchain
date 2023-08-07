from __future__ import annotations

from typing import Any, Sequence

from langchain.automaton.automaton import ExecutedState, State, Automaton
from langchain.automaton.typedefs import (
    MessageType,
    infer_message_type,
    PromptGenerator,
)
from langchain.automaton.well_known_states import LLMProgram, UserInputState
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


class ChatAutomaton(Automaton):
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        prompt_generator: PromptGenerator,
    ) -> None:
        """Initialize the chat automaton."""
        super().__init__()
        self.llm = llm
        self.tools = tools
        # TODO: Fix mutability of chat template, potentially add factory method
        self.prompt_generator = prompt_generator
        self.llm_program_state = LLMProgram(
            llm=self.llm,
            tools=self.tools,
            prompt_generator=self.prompt_generator,
        )

    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        """Get the start state."""
        return self.llm_program_state

    def get_next_state(
        self, executed_state: ExecutedState  # Add memory for transition functions?
    ) -> State:
        """Get the next state."""
        previous_state_id = executed_state["id"]
        data = executed_state["data"]

        if previous_state_id == "user_input":
            return self.llm_program_state
        elif previous_state_id == "llm_program":
            message_type = infer_message_type(data["message"])
            if message_type == MessageType.AI:
                return UserInputState()
            elif message_type == MessageType.FUNCTION:
                return self.llm_program_state
            else:
                raise AssertionError(f"Unknown message type: {message_type}")
        else:
            raise ValueError(f"Unknown state ID: {previous_state_id}")


# This is transition matrix syntax
# transition_matrix = {
#     ("user_input", "*"): LLMProgram,
#     ("llm_program", MessageType.AI): UserInputState,
#     ("llm_program", MessageType.AI_SELF): LLMProgram,
#     (
#         "llm_program",
#         MessageType.AI_INVOKE,
#     ): FuncInvocationState,  # But must add function message
#     ("func_invocation", MessageType.FUNCTION): LLMProgram,
# }
#
