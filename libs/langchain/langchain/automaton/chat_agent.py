"""Generalized chat agent, works with any chat model."""
from __future__ import annotations

from typing import Callable, Iterator, Optional, Sequence, TypeVar, Union, List, Mapping

from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    Agent,
    AgentFinish,
    MessageLike,
)
from langchain.schema import PromptValue
from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelInput,
    LanguageModelOutput,
)
from langchain.automaton.processors import WorkingMemoryManager
from langchain.schema.messages import BaseMessage
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
)
from langchain.tools import BaseTool

T = TypeVar("T")


class ChatAgent(Agent):
    """A generalized chat agent."""

    def __init__(
        self,
        llm: BaseLanguageModel[LanguageModelInput, LanguageModelOutput]
        | Runnable[LanguageModelInput, LanguageModelOutput],
        prompt_generator: Union[
            Callable[
                [Sequence[MessageLike]], Union[str, PromptValue, List[BaseMessage]]
            ],
            Runnable,
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
        memory_processor: Optional[WorkingMemoryManager] = None,
    ) -> None:
        """Initialize the chat agent."""
        invoke_tools = bool(tools)
        self.prompt_generator = prompt_generator
        self.llm_program: Runnable[
            Sequence[MessageLike], List[MessageLike]
        ] = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=parser,
            stop=stop,
            invoke_tools=invoke_tools,
        )
        self.memory_processor = memory_processor

    def step(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[MessageLike]:
        """Implement a single step of the agent."""
        last_message = messages[-1] if messages else None
        if not last_message:
            return []

        new_messages = self.memory_processor.process(messages)

        match last_message:
            case AgentFinish():
                return []
            case _:
                return self.llm_program.invoke(new_messages, config=config)


WorkingMemoryProcessor = Runnable[Sequence[MessageLike], List[MessageLike]]
Router = Callable[[Sequence[MessageLike], ...], Optional[WorkingMemoryProcessor]]


class SimpleAutomaton:
    def __init__(
        self,
        router: Callable[[Sequence[MessageLike]], Optional[WorkingMemoryProcessor]],
    ) -> None:
        """Initialize the automaton."""
        self.router = router

    def run(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[RunnableConfig] = None,
        max_iterations: int = 10,
    ) -> List[MessageLike]:
        """Run the automaton."""
        new_messages = list(messages)
        for _ in range(max_iterations):
            runnable = self.router(new_messages)
            if not runnable:
                break
            new_messages.extend(runnable.invoke(new_messages, config=config))
        return new_messages
