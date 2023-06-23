"""Base classes for comparing the output of two models."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import Field

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.evaluation.comparison.prompt import PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser


@runtime_checkable
class PairwiseStringEvaluator(Protocol):
    """A protocol for evaluating the output of two models."""

    @abstractmethod
    def evaluate_string_pairs(
        self, *, output_a: str, output_b: str, input: str, **kwargs: Any
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            output_a: The output string from the first model.
            output_b: The output string from the second model.
            input: The input or task string.

        Returns:
            A dictionary containing the preference, scores, and/or
            other information.
        """


class PairwiseResultOutputParser(BaseOutputParser[dict]):
    def parse(self, text: str) -> Any:
        reasoning, verdict = text.strip().rsplit("\n", maxsplit=1)
        verdict = verdict.strip("[").strip("]")
        return {
            "comment": reasoning,
            "value": verdict,
        }


class PairwiseStringEvalChain(LLMChain):
    """A chain for comparing the output of two models."""

    output_parser: BaseOutputParser = Field(default_factory=PairwiseResultOutputParser)

    @classmethod
    def from_llm(
        cls, *, llm: BaseLanguageModel, prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> PairwiseStringEvalChain:
        """Initialize the PairwiseStringEvalChain from an LLM.

        Args:
            llm: The LLM to use.
            prompt: The prompt to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The initialized PairwiseStringEvalChain.
        """
        expected_input_vars = {"output_a", "output_b", "input"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate_string_pairs(
        self,
        *,
        output_a: str,
        output_b: str,
        input: str,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            output_a: The output string from the first model.
            output_b: The output string from the second model.
            input: The input or task string.
            callbacks: The callbacks to use.

        Returns:
            A dictionary containing the preference, scores, and/or
            other information.
        """
        result = self(
            {
                "output_a": output_a,
                "output_b": output_b,
                "input": input,
            },
            callbacks=callbacks,
            **kwargs,
        )
        return result["text"]
