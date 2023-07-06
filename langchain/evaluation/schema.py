"""Interfaces to be implemented by general evaluators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple

from langchain.schema.agent import AgentAction


class StringEvaluator(ABC):
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


class PairwiseStringEvaluator(ABC):
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


class AgentTrajectoryEvaluator(ABC):
    """Interface for evaluating agent trajectories."""

    @abstractmethod
    def evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            input (str): The input to the agent.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """

    async def aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            input (str): The input to the agent.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} hasn't implemented an async "
            "aevaluate_agent_trajectory method."
        )
