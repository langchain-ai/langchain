"""An automaton."""
from __future__ import annotations

import abc
import dataclasses
import enum
from typing import (
    Sequence,
    Callable,
    Any,
    Mapping,
    TypedDict,
    Protocol,
)

from langchain.automaton.open_ai_functions import create_action_taking_llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema import (
    BaseMessage,
    FunctionMessage,
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseChatMessageHistory,
    PromptValue,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


class MessageType(enum.Enum):
    """The type of message."""

    SYSTEM = enum.auto()
    USER = enum.auto()
    FUNCTION = enum.auto()
    AI = enum.auto()
    AI_INVOKE = enum.auto()


def _infer_message_type(message: BaseMessage) -> MessageType:
    """Assign message type."""
    if isinstance(message, FunctionMessage):
        return MessageType.FUNCTION
    elif isinstance(message, AIMessage):
        if message.additional_kwargs:
            return MessageType.AI_INVOKE
        else:
            return MessageType.AI
    elif isinstance(message, SystemMessage):
        return MessageType.SYSTEM
    elif isinstance(message, HumanMessage):
        return MessageType.USER
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


class ExecutedState(TypedDict):
    """The response of an action taking LLM."""

    id: str  # the ID of the state that was just executed
    data: Mapping[str, Any]


class State(Protocol):
    """Automaton state protocol."""

    def execute(self) -> ExecutedState:
        ...


class Memory(BaseChatMessageHistory):
    """A memory for the automaton."""

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the memory."""
        self.messages.append(message)


PromptGenerator = Callable[[Memory], PromptValue]


@dataclasses.dataclass
class LLMProgram(State):
    """A state that executes an LLM program."""

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    # # This should either be swapped with memory or else with prompt value?
    # # Likely prompt value since we're not taking in any input
    # messages: Sequence[BaseMessage]  # Swap with prompt value
    memory: Memory
    prompt_generator: PromptGenerator

    def execute(self) -> ExecutedState:
        """Execute LLM program."""
        action_taking_llm = create_action_taking_llm(self.llm, tools=self.tools)
        prompt_value = self.prompt_generator(self.memory)

        result = action_taking_llm.invoke(prompt_value)
        self.memory.add_message(result["message"])
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


class Automaton:
    @abc.abstractmethod
    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_state(self, executed_state: ExecutedState) -> State:
        raise NotImplementedError()


class ChatAutomaton(Automaton):
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        on_next_state: Callable[[State], None],
    ) -> None:
        """Initialize the chat automaton."""
        self.llm = llm
        self.tools = tools
        # TODO: Fix mutability of chat template, potentially add factory method
        self.chat_template = ChatPromptTemplate.from_messages(prompt.format_messages())
        self.on_next_state = on_next_state

    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        """Get the start state."""
        return LLMProgram(
            llm=self.llm,
            tools=self.tools,
            messages=self.chat_template.format_messages(),
        )

    def get_next_state(self, executed_state: ExecutedState) -> State:
        """Get the next state."""
        self.on_next_state(executed_state)
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
            message_type = _infer_message_type(data["message"])
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
