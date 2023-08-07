from __future__ import annotations

import dataclasses
import json
from typing import (
    Any,
    List,
    Sequence,
    Mapping,
    TypeVar,
    Iterator,
    Tuple,
)

from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from langchain.automaton.automaton import (
    _infer_message_type,
    ExecutedState,
    State,
    Automaton,
    ChatAutomaton,
)

from langchain.automaton.open_ai_functions import (
    OpenAIFunctionsRouter,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import ChatResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
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


def test_openai_functions_router() -> None:
    """Test the OpenAIFunctionsRouter."""
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


def print_message(message: BaseMessage) -> None:
    """Pretty print a message."""
    print(f"{_infer_message_type(message).name}: {message.content}")


T = TypeVar("T")


@dataclasses.dataclass
class FunctionInvocation(State):
    llm: BaseLanguageModel
    tools: Sequence[BaseTool]


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
    state, executed_states = executor.run()
    assert executed_states == [
        {
            "data": {
                "function_call": {
                    "arguments": {},
                    "name": "get_time",
                    "result": "9 PM",
                },
                "message": AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {"name": "get_time", "arguments": "{}"}
                    },
                    example=False,
                ),
            },
            "id": "llm_program",
        }
    ]
