from __future__ import annotations

import abc
import enum
import json
from typing import Any, List, Sequence, Mapping, TypedDict, TypeVar, Generic, Iterator

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
    def __init__(self, messages: Sequence[BaseMessage], **kwargs: Any) -> None:
        """Initialize the model."""
        self.response_iter = iter(messages)
        super().__init__(**kwargs)

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
        message = next(self.response_iter)
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
        messages=[
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


class State(Generic[T]):
    """A state in the automaton."""

    @abc.abstractmethod
    def execute(self) -> T:
        """Execute the state."""


class Automaton(Generic[T]):
    @abc.abstractmethod
    def get_start_state(self, *args, **kwargs) -> State[T]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_state(self, *args, **kwargs) -> State[T]:
        raise NotImplementedError()


class ActionTakingResponse(TypedDict):
    """The response of an action taking LLM."""

    message: BaseMessage


class LLMProgramState(State):
    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    messages: Sequence[BaseMessage]

    def execute(self) -> T:
        """Execute LLM program."""
        action_taking_llm = create_action_taking_llm(self.llm, self.tools)
        result = action_taking_llm.invoke(self.messages)
        return {
            "llm": self.llm,
            "tools": self.tools,
        }


class UserInputState(State):
    def execute(self) -> T:
        """Execute user input state."""
        user_input = input("Enter your input: ")
        return {
            "type_": "user_input",
        }


class Executor:
    def __init__(self, automaton: Automaton) -> None:
        """Initialize the executor."""
        self.automaton = automaton

    def run(self) -> None:
        """Run the automaton."""
        state = self.automaton.get_start_state()

        for _ in range(10):
            new_state = state.execute()

        result = state.execute()
        new_state = state.execute()


@tool_maker
def get_time() -> str:
    """Get time."""
    return "9 PM"


@tool_maker
def get_location() -> str:
    """Get location."""
    return "the park"


def run_automaton() -> None:
    """Run the automaton."""
    tools = [get_time, get_location]
    llm = FakeChatOpenAI(
        messages=[
            _construct_func_invocation_message(get_time, {}),
            AIMessage(
                content="The time is 9 PM.",
            ),
        ]
    )
    chain = create_action_taking_llm(llm, tools=tools)
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant cat. Please use the tools at "
                "your disposal to help the human. "
                "You can ask the user to clarify if need be.",
            ),
        ]
    )

    last_response = {
        "data": None,
        "name": None,
        "last_message": template.format_messages()[-1],
    }

    for _ in range(1):
        last_message = template.format_messages()[-1]
        message_type = _infer_message_type(last_message)

        if message_type.AI_INVOKE:
            pass
        elif message_type.AI:
            # Then transition to user
            pass
        elif message_type.USER:
            # (Ready for human input?)
            # Determine if human turn
            content = input("User:")
            if content == "q":  # Quit
                break
            template.append(("human", content))
            # then transition to AI
            pass
        elif message_type.SYSTEM:
            # then transition to AI or User
            pass
        elif message_type.FUNCTION:
            # then transition to AI
            pass

        if last_response and last_response["name"] == "bye":
            print("AGI: byebye silly human")
            break

        # Very hacky routing layer
        if isinstance(last_message, SystemMessage) or (
            (last_message, AIMessage) and not last_message.additional_kwargs
        ):  # (Ready for human input?)
            # Determine if human turn
            content = input("User:")
            if content == "q":  # Quit
                break
            template = template + [("human", content)]
        else:  # Determine if need to insert function invocation information
            if (
                last_response and last_response["name"]
            ):  # Last response was a tool invocation, need to append a Function message
                template.append(
                    FunctionMessage(
                        content=last_response["data"], name=last_response["name"]
                    )
                )
                print_message(template.messages[-1])

        messages = template.format_messages()  # Would love to get rid of this

        last_response = chain.invoke(messages)
        template.append(last_response["message"])


def test_automaton() -> None:
    """Test the automaton by running a simple chat model."""
    run_automaton()
