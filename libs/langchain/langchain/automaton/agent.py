"""Generalized chat agent, works with any chat model."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar, List

from langchain.automaton.typedefs import MessageLike, AgentFinish, Agent
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
)

T = TypeVar("T")


WorkingMemoryProcessor = Runnable[Sequence[MessageLike], List[MessageLike]]
LLMProgram = Runnable[Sequence[MessageLike], List[MessageLike]]


class SequentialAgent(Agent):
    def __init__(
        self,
        llm_program: LLMProgram,
        memory_processor: Optional[WorkingMemoryProcessor] = None,
    ) -> None:
        """Initialize the agent."""
        self.llm_program = llm_program
        self.memory_processor = memory_processor

    def run(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[RunnableConfig] = None,
        max_iterations: int = 10,
    ) -> List[MessageLike]:
        """Run the agent."""
        messages = list(messages)
        for iteration_num in range(max_iterations):
            # Working memory / working state updates can take the form
            # of appends or replacements
            if self.memory_processor:
                # This is a replacement
                messages = self.memory_processor.invoke(messages)
            if messages and isinstance(messages[-1], AgentFinish):
                break

            # This is an append to working memory
            messages.extend(self.llm_program.invoke(messages, config=config))

        return messages


class MessageAutomaton:  # Just a sketch
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
