from __future__ import annotations

from typing import Any, Sequence

from langchain.automaton.automaton import ExecutedState, State, Automaton
from langchain.automaton.typedefs import MessageType, infer_message_type
from langchain.automaton.well_known_states import LLMProgram, UserInputState
from langchain.prompts import ChatPromptTemplate
from langchain.schema import (
    FunctionMessage,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


class ChatAutomaton(Automaton):
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
    ) -> None:
        """Initialize the chat automaton."""
        self.llm = llm
        self.tools = tools
        # TODO: Fix mutability of chat template, potentially add factory method
        self.chat_template = ChatPromptTemplate.from_messages(prompt.format_messages())

    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        """Get the start state."""
        return LLMProgram(
            llm=self.llm,
            tools=self.tools,
            messages=self.chat_template.format_messages(),
        )

    def get_next_state(self, executed_state: ExecutedState) -> State:
        """Get the next state."""
        previous_state_id = executed_state["id"]
        data = executed_state["data"]
        self.chat_template.append(data["message"])

        if previous_state_id == "user_input":
            return LLMProgram(
                llm=self.llm,
                tools=self.tools,
                # Could add memory here
                messages=self.chat_template.format_messages(),
            )
        elif previous_state_id == "llm_program":
            message_type = infer_message_type(data["message"])
            if message_type in {MessageType.USER, MessageType.FUNCTION}:
                raise AssertionError(
                    "LLM program should not return user or function messages."
                )
            elif message_type == MessageType.AI:
                return UserInputState()
            elif message_type == MessageType.AI_INVOKE:
                # Here we need to add a function message
                # and then return the user input state.
                assert data["function_call"]

                function_message = FunctionMessage(
                    name=data["function_call"]["name"],
                    content=data["function_call"]["result"],
                )

                # Function message requires custom addition
                # Logic may need to be refactored
                self.chat_template.append(function_message)

                return LLMProgram(
                    llm=self.llm,
                    tools=self.tools,
                    messages=self.chat_template.format_messages(),
                )
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
