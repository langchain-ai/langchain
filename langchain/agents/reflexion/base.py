from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseOutputParser


class ReflexionOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> str:
        """Parse text into agent action/finish."""


class BaseReflector(BaseModel):
    """Agent for the Reflexer chain."""

    llm_chain: LLMChain
    output_parser: ReflexionOutputParser

    max_iterations_per_trial: Optional[int] = 15
    max_execution_time_per_trial: Optional[float] = None

    trial_history: List[str] = []
    """Full string representation of each trial"""
    trial_reflexions: List[str] = []
    """Reflexion for of each trial"""
    trial_prefix: str = "\nTrial {trial_number}"
    trial_suffix: str = "\nSTATUS: FAIL\nNew plan: "

    def current_trial_prefix(self, trial_number: int) -> str:
        return self.trial_prefix.replace("{trial_number}", str(trial_number))

    @abstractmethod
    def get_history(self, trials: int) -> str:
        """Return reflexion history, so it can be used in agent execution prompt"""

    @classmethod
    @abstractmethod
    def create_prompt(self) -> BasePromptTemplate:
        """Prompt to pass to LLM."""

    def should_reflect(self, iterations_in_trial: int,
                       execution_time_in_trial: float,
                       *args: Any, **kwargs: Any) -> bool:
        """Determine if we should reflect, e.g. when current trial failed."""
        # We reflect when ...
        # ... we have too many iterations in current trial, or
        if (self.max_iterations_per_trial is not None
            and iterations_in_trial >= self.max_iterations_per_trial):
            return True
        # ... current trial took too long
        if (self.max_execution_time_per_trial is not None
            and execution_time_in_trial >= self.max_execution_time_per_trial):
            return True
        return False

    @abstractmethod
    def reflect(
            self,
            input: str,
            current_trial: str,
            current_trial_no: int,
            callbacks: Callbacks = None) -> str:
        """ returns full relection notes  """

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[ReflexionOutputParser] = None,
        **kwargs: Any,
    ) -> BaseReflector:
        """Construct a reflector from an LLM."""
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(),
            callback_manager=callback_manager,
        )
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            output_parser=_output_parser,
            **kwargs,
        )

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> ReflexionOutputParser:
        """Get default output parser for this class."""
