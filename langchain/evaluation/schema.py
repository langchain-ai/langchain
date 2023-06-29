"""Interfaces to be implemented by general evaluators."""
from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain


class EvaluatorType(str, Enum):
    """The types of the evaluators."""

    QA = "qa"
    """Question answering evaluator, which grades answers to questions
    directly using an LLM."""
    COT_QA = "cot_qa"
    """Chain of thought question answering evaluator, which grades
    answers to questions using
    chain of thought 'reasoning'."""
    CONTEXT_QA = "context_qa"
    """Question answering evaluator that incorporates 'context' in the response."""
    PAIRWISE_STRING = "pairwise_string"
    """The pairwise string evaluator, which compares the output of two models."""
    AGENT_TRAJECTORY = "trajectory"
    """The agent trajectory evaluator, which grades the agent's intermediate steps."""
    CRITERIA = "criteria"
    """The criteria evaluator, which evaluates a model based on a
    custom set of criteria."""


class EvalChain(Chain):
    """A base class for evaluators that use an LLM."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> EvalChain:
        """Create a new evaluator from an LLM."""
        return cls(llm=llm, **kwargs)


@runtime_checkable
class StringEvaluator(Protocol):
    """Protocol for evaluating strings."""

    @abstractmethod
    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): the LLM or chain prediction to evaluate.
            reference (Optional[str], optional): the reference label
                to evaluate against.
            input (Optional[str], optional): the input to consider during evaluation
            **kwargs: additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate Chain or LLM output, based on optional
          input and label.

        Args:
            prediction (str): the LLM or chain prediction to evaluate.
            reference (Optional[str], optional): the reference label
                 to evaluate against.
            input (Optional[str], optional): the input to consider during evaluation
            **kwargs: additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an "
            "async aevaluate_strings method."
        )


@runtime_checkable
class PairwiseStringEvaluator(Protocol):
    """A protocol for comparing the output of two models."""

    @abstractmethod
    def evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (str, optional): The expected output / reference
                string. Defaults to None.
            input (str, optional): The input string. Defaults to None.
            **kwargs (Any): Additional keyword arguments, such
                as callbacks and optional reference strings.

        Returns:
            dict: A dictionary containing the preference, scores, and/or
                other information.
        """

    async def aevaluate_string_pairs(
        self,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (str, optional): The expected output / reference
                string. Defaults to None.
            input (str, optional): The input string. Defaults to None.
            **kwargs (Any): Additional keyword arguments, such
                as callbacks and optional reference strings.

        Returns:
            dict: A dictionary containing the preference, scores, and/or
                other information.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an async "
            "aevaluate_string_pairs method."
        )
