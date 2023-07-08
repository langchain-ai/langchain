"""Interfaces to be implemented by general evaluators."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple
from warnings import warn

from langchain.chains.base import Chain
from langchain.schema.agent import AgentAction
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)


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


class LLMEvalChain(Chain):
    """A base class for evaluators that use an LLM."""

    @classmethod
    @abstractmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> LLMEvalChain:
        """Create a new evaluator from an LLM."""


class _EvalArgsMixin:
    """Mixin for checking evaluation arguments."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input string."""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """Warning to show when input is ignored."""
        return f"Ignoring input in {self.__class__.__name__}, as it is not expected."

    @property
    def _skip_reference_warning(self) -> str:
        """Warning to show when reference is ignored."""
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
        )

    def _check_evaluation_args(
        self,
        reference: Optional[str] = None,
        input: Optional[str] = None,
    ) -> None:
        if self.requires_input and input is None:
            raise ValueError(f"{self.__class__.__name__} requires an input string.")
        elif input is not None and not self.requires_input:
            warn(self._skip_input_warning)
        else:
            pass
        if self.requires_reference and reference is None:
            raise ValueError(f"{self.__class__.__name__} requires a reference string.")
        elif reference is not None and not self.requires_reference:
            warn(self._skip_reference_warning)
        else:
            pass


class StringEvaluator(_EvalArgsMixin, ABC):
    """Protocol for evaluating strings."""

    @property
    def evaluation_name(self) -> str:
        raise NotImplementedError()

    @property
    def requires_reference(self) -> bool:
        return False

    @abstractmethod
    def _evaluate_strings(
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
                It is recommended that the dictionary contain the following keys:
                    - score: the score of the evaluation, if applicable.
                    - value: the string value of the evaluation, if applicable.
                    - reasoning: the reasoning for the evaluation, if applicable.
        """

    async def _aevaluate_strings(
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
                It is recommended that the dictionary contain the following keys:
                    - score: the score of the evaluation, if applicable.
                    - value: the string value of the evaluation, if applicable.
                    - reasoning: the reasoning for the evaluation, if applicable.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an "
            "async aevaluate_strings method."
        )

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
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )

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
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )


class PairwiseStringEvaluator(_EvalArgsMixin, ABC):
    """A protocol for comparing the output of two models."""

    @abstractmethod
    def _evaluate_string_pairs(
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

    async def _aevaluate_string_pairs(
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
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an async "
            "aevaluate_string_pairs method."
        )

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
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )

    async def aevaluate_string_pairs(
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
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )


class AgentTrajectoryEvaluator(_EvalArgsMixin, ABC):
    """Interface for evaluating agent trajectories."""

    @property
    def requires_input(self) -> bool:
        return True

    @abstractmethod
    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        input: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            input (str): The input to the agent.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """

    async def _aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        input: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            input (str): The input to the agent.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an async "
            "aevaluate_agent_trajectory method."
        )

    def evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        input: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            input (str): The input to the agent.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate_agent_trajectory(
            prediction=prediction,
            input=input,
            agent_trajectory=agent_trajectory,
            reference=reference,
            **kwargs,
        )

    async def aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        input: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            input (str): The input to the agent.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_agent_trajectory(
            prediction=prediction,
            input=input,
            agent_trajectory=agent_trajectory,
            reference=reference,
            **kwargs,
        )
