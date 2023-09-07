"""Generalized chat agent, works with any chat model."""
from __future__ import annotations

from typing import TypeVar, Callable, Optional, Sequence, Union, Iterator

from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    AgentFinish,
    MessageLike,
    Agent,
)
from langchain.schema import PromptValue
from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelOutput,
    LanguageModelInput,
)
from langchain.schema.messages import BaseMessage
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.runnable import (
    Runnable,
)
from langchain.tools import BaseTool

T = TypeVar("T")



class ChatAgent(Agent):
    """A generalized chat agent."""

    def __init__(
        self,
        llm: BaseLanguageModel[LanguageModelInput, LanguageModelOutput],
        prompt_generator: Union[
            Callable[[T], Union[str, PromptValue, Sequence[BaseMessage]]], Runnable
        ],
        *,
        tools: Optional[Sequence[BaseTool]] = None,
        stop: Optional[Sequence[str]] = None,
        parser: Union[
            Runnable[Union[BaseMessage, str], MessageLike],
            Callable[[Union[BaseMessage, str]], MessageLike],
            BaseOutputParser,
            None,
        ] = None,
    ) -> None:
        """Initialize the chat agent."""
        invoke_tools = bool(tools)
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=parser,
            stop=stop,
            invoke_tools=invoke_tools,
        )

    def run(
        self,
        messages: Sequence[MessageLike],
        *,
        max_iterations: int = 100,
    ) -> Iterator[MessageLike]:
        """Run the agent."""
        all_messages = list(messages)
        for _ in range(max_iterations):
            if all_messages and isinstance(all_messages[-1], AgentFinish):
                break
            new_messages = self.llm_program.invoke(all_messages)
            yield from new_messages
            all_messages.extend(new_messages)
