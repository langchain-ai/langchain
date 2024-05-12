from __future__ import annotations

from textwrap import indent
from typing import Any, Dict, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain_experimental.pydantic_v1 import Extra
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.controller import ToTController
from langchain_experimental.tot.memory import ToTDFSMemory
from langchain_experimental.tot.thought import Thought, ThoughtValidity
from langchain_experimental.tot.thought_generation import (
    BaseThoughtGenerationStrategy,
    ProposePromptStrategy,
)


class ToTChain(Chain):
    """
    Chain implementing the Tree of Thought (ToT).
    """

    llm: BaseLanguageModel
    """
    Language model to use. It must be set to produce different variations for
    the same prompt.
    """
    checker: ToTChecker
    """ToT Checker to use."""
    output_key: str = "response"  #: :meta private:
    k: int = 10
    """The maximum number of conversation rounds"""
    c: int = 3
    """The number of children to explore at each node"""
    tot_memory: ToTDFSMemory = ToTDFSMemory()
    tot_controller: ToTController = ToTController()
    tot_strategy_class: Type[BaseThoughtGenerationStrategy] = ProposePromptStrategy
    verbose_llm: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> ToTChain:
        """
        Create a ToTChain from a language model.

        :param llm: The language model to use.
        :param kwargs: Additional arguments to pass to the ToTChain constructor.
        """
        return cls(llm=llm, **kwargs)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tot_controller.c = self.c

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def log_thought(
        self,
        thought: Thought,
        level: int,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> None:
        if run_manager:
            colors = {
                ThoughtValidity.VALID_FINAL: "green",
                ThoughtValidity.VALID_INTERMEDIATE: "yellow",
                ThoughtValidity.INVALID: "red",
            }
            text = indent(f"Thought: {thought.text}\n", prefix="    " * level)
            run_manager.on_text(
                text=text, color=colors[thought.validity], verbose=self.verbose
            )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        if run_manager:
            run_manager.on_text(text="Starting the ToT solve procedure.\n")

        problem_description = inputs["problem_description"]
        checker_inputs = {"problem_description": problem_description}
        thoughts_path: tuple[str, ...] = ()
        thought_generator = self.tot_strategy_class(
            llm=self.llm, c=self.c, verbose=self.verbose_llm
        )

        level = 0
        for _ in range(self.k):
            level = self.tot_memory.level
            thought_text = thought_generator.next_thought(
                problem_description, thoughts_path, callbacks=_run_manager.get_child()
            )
            checker_inputs["thoughts"] = thoughts_path + (thought_text,)
            thought_validity = self.checker(
                checker_inputs, callbacks=_run_manager.get_child()
            )["validity"]
            thought = Thought(text=thought_text, validity=thought_validity)
            if thought.validity == ThoughtValidity.VALID_FINAL:
                self.log_thought(thought, level, run_manager)
                return {self.output_key: thought.text}
            self.tot_memory.store(thought)
            self.log_thought(thought, level, run_manager)
            thoughts_path = self.tot_controller(self.tot_memory)

        return {self.output_key: "No solution found"}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented yet")

    @property
    def _chain_type(self) -> str:
        return "tot"
