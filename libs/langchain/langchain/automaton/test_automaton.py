from __future__ import annotations

import json
from typing import (
    Any,
    List,
    Mapping,
    Iterator,
    Sequence,
)

from langchain.automaton.chat_automaton import ChatAutomaton
from langchain.automaton.executor import Executor
from langchain.automaton.open_ai_functions import (
    OpenAIFunctionsRouter,
)
from langchain.automaton.typedefs import Memory, PromptGenerator
from langchain.schema import PromptValue
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
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


class EasyPromptValue(PromptValue):
    messages: List[BaseMessage]

    @classmethod
    def from_messages(cls, messages: Sequence[BaseMessage]) -> EasyPromptValue:
        return cls(messages=messages)

    def to_messages(self) -> List[BaseMessage]:
        return self.messages

    def to_string(self) -> str:
        return " ".join([message.content for message in self.messages])


def prompt_generator(memory: Memory) -> PromptValue:
    """Generate a prompt."""
    if not memory.messages:
        memory.add_message(
            SystemMessage(
                content=(
                    "Hello! I'm a chatbot that can help you write a letter. "
                    "What would you like to do?"
                ),
            )
        )

    return EasyPromptValue.from_messages(messages=memory.messages)


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

    # TODO(FIX MUTABILITY)
    chat_automaton = ChatAutomaton(
        llm=llm, tools=tools, prompt_generator=prompt_generator
    )
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
