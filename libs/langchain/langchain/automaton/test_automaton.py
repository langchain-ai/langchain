from __future__ import annotations

import abc
import dataclasses
import enum
import json
from typing import Any, List, Sequence, Mapping, TypedDict, TypeVar, Iterator, Tuple

from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langchain.automaton.open_ai_functions import (
    OpenAIFunctionsRouter,
    create_action_taking_llm,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import ChatResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
)
from langchain.schema.output import ChatGeneration
from langchain.schema.runnable import RunnableLambda
from langchain.tools.base import tool as tool_maker, BaseTool


class FakeChatOpenAI(BaseChatModel):
    """A fake chat model that returns a pre-defined response."""

    message_iter: Iterator[BaseMessage]

    @property
    def _llm_type(self) -> str:
        """The type of the model."""
        return "fake-openai-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response to the given messages."""
        message = next(self.message_iter)
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_openai_functions_router(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    def revise(notes: str) -> str:
        """Revises the draft."""
        return f"Revised draft: {notes}!"

    def accept(draft: str) -> str:
        """Accepts the draft."""
        return f"Accepted draft: {draft}!"

    router = OpenAIFunctionsRouter(
        functions=[
            {
                "name": "revise",
                "description": "Sends the draft for revision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "The editor's notes to guide the revision.",
                        },
                    },
                },
            },
            {
                "name": "accept",
                "description": "Accepts the draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "string",
                            "description": "The draft to accept.",
                        },
                    },
                },
            },
        ],
        runnables={
            "revise": RunnableLambda(lambda x: revise(x["revise"])),
            "accept": RunnableLambda(lambda x: accept(x["draft"])),
        },
    )

    model = FakeChatOpenAI(
        message_iter=iter(
            [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "accept",
                            "arguments": '{\n  "draft": "turtles"\n}',
                        }
                    },
                )
            ]
        )
    )

    chain = model.bind(functions=router.functions) | router

    assert chain.invoke("Something about turtles?") == "Accepted draft: turtles!"


def _construct_func_invocation_message(
    tool: BaseTool, args: Mapping[str, Any]
) -> AIMessage:
    """Construct a function invocation message."""
    return AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": tool.name,
                "arguments": json.dumps(args),
            }
        },
    )


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


def print_message(message: BaseMessage) -> None:
    """Pretty print a message."""
    print(f"{_infer_message_type(message).name}: {message.content}")


T = TypeVar("T")


from typing import Mapping


class ExecutedState(TypedDict):
    """The response of an action taking LLM."""

    id_: str  # the ID of the state that was just executed
    data: Mapping[str, Any]


@dataclasses.dataclass
class State:
    """A state in the automaton."""

    @abc.abstractmethod
    def execute(self) -> ExecutedState:
        """Execute the state."""


@dataclasses.dataclass
class LLMProgramState(State):
    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    messages: Sequence[BaseMessage]

    def execute(self) -> ExecutedState:
        """Execute LLM program."""
        action_taking_llm = create_action_taking_llm(self.llm, tools=self.tools)
        result = action_taking_llm.invoke(self.messages)
        return {"id_": "llm_program", "data": result}


@dataclasses.dataclass
class UserInputState(State):
    def execute(self) -> ExecutedState:
        """Execute user input state."""
        user_input = input("Enter your input: ")
        return {
            "id_": "user_input",
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
    ) -> None:
        """Initialize the chat automaton."""
        self.llm = llm
        self.tools = tools
        # TODO: Fix mutability of chat template, potentially add factory method
        self.chat_template = ChatPromptTemplate.from_messages(prompt.format_messages())

    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        """Get the start state."""
        return LLMProgramState(
            llm=self.llm,
            tools=self.tools,
            messages=self.chat_template.format_messages(),
        )

    def get_next_state(self, executed_state: ExecutedState) -> State:
        """Get the next state."""
        previous_state_id = executed_state["id_"]
        data = executed_state["data"]
        self.chat_template.append(data["message"])

        if previous_state_id == "user_input":
            return LLMProgramState(
                llm=self.llm,
                tools=self.tools,
                messages=self.chat_template.format_messages(),
            )
        elif previous_state_id == "llm_program":
            message_type = _infer_message_type(data["message"])
            if message_type == MessageType.USER:
                raise AssertionError(
                    "LLM program should not return user input message."
                )
            elif message_type == MessageType.FUNCTION:
                raise AssertionError(
                    "User input state should not return function message."
                )
            elif message_type in MessageType.AI:
                return UserInputState()
            elif message_type == MessageType.AI_INVOKE:
                # Here we need to add a function message
                # and then return the user input state.
                assert data["function_call"]

                function_message = FunctionMessage(
                    content=data["function_call"]["result"],
                )

                # Function message requires custom addition
                # Logic may need to be refactored
                self.chat_template.append(function_message)

                return LLMProgramState(
                    llm=self.llm,
                    tools=self.tools,
                    messages=self.chat_template.format_messages(),
                )

        else:
            raise ValueError(f"Unknown state ID: {previous_state_id}")


# Need to make into runnable
# This is a for looping runnable... :)
class Executor:
    def __init__(self, automaton: Automaton, max_iterations: int) -> None:
        """Initialize the executor."""
        self.automaton = automaton
        self.max_iterations = max_iterations

    def run(self) -> Tuple[State, List[ExecutedState]]:
        """Run the automaton.

        Returns:
            The final state and result of executed states.
        """
        state = self.automaton.get_start_state()
        executed_states = []

        for _ in range(self.max_iterations):
            raise ValueError(state)
            executed_state = state.execute()
            raise ValueError(executed_state)
            executed_states.append(executed_state)
            state = self.automaton.get_next_state(executed_state)

        return state, executed_states


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
                _construct_func_invocation_message(get_time, {}),
                AIMessage(
                    content="The time is 9 PM.",
                ),
            ]
        )
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant cat. Please use the tools at "
                "your disposal to help the human. "
                "You can ask the user to clarify if need be.",
            ),
        ]
    )

    # TODO(FIX MUTABILITY)
    chat_automaton = ChatAutomaton(llm=llm, tools=tools, prompt=prompt)
    executor = Executor(chat_automaton, max_iterations=1)
    executed_states = executor.run()
    assert executed_states == []
