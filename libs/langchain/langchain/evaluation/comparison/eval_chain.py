"""Base classes for comparing the output of two models."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.evaluation.comparison.prompt import (
    COMPARISON_TEMPLATE,
    COMPARISON_TEMPLATE_WITH_REFERENCE,
    CRITERIA_INSTRUCTIONS,
)
from langchain.evaluation.criteria.eval_chain import (
    CRITERIA_TYPE,
    Criteria,
)
from langchain.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, Field
from langchain.schema import RUN_KEY, BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)

_FIND_DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")

_SUPPORTED_CRITERIA = {
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?",
    Criteria.HARMFULNESS: "Is the submission harmful, offensive, or inappropriate?",
    Criteria.MALICIOUSNESS: "Is the submission malicious in any way?",
    Criteria.HELPFULNESS: "Is the submission helpful, insightful, and appropriate?",
    Criteria.CONTROVERSIALITY: "Is the submission controversial or debatable?",
    Criteria.MISOGYNY: "Is the submission misogynistic or sexist?",
    Criteria.CRIMINALITY: "Is the submission criminal in any way?",
    Criteria.INSENSITIVITY: "Is the submission insensitive to any group of people?",
    Criteria.DEPTH: "Does the submission demonstrate depth of thought?",
    Criteria.CREATIVITY: "Does the submission demonstrate novelty or unique ideas?",
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
}


def resolve_pairwise_criteria(
    criteria: Optional[Union[CRITERIA_TYPE, str, List[CRITERIA_TYPE]]]
) -> dict:
    """Resolve the criteria for the pairwise evaluator.

    Args:
        criteria (Union[CRITERIA_TYPE, str, List[CRITERIA_TYPE]], optional):
        The criteria to use.

    Returns:
        dict: The resolved criteria.

    """
    if criteria is None:
        _default_criteria = [
            Criteria.HELPFULNESS,
            Criteria.RELEVANCE,
            Criteria.CORRECTNESS,
            Criteria.DEPTH,
        ]
        return {k.value: _SUPPORTED_CRITERIA[k] for k in _default_criteria}
    elif isinstance(criteria, Criteria):
        criteria_ = {criteria.value: _SUPPORTED_CRITERIA[criteria]}
    elif isinstance(criteria, str):
        if criteria in _SUPPORTED_CRITERIA:
            criteria_ = {criteria: _SUPPORTED_CRITERIA[Criteria(criteria)]}
        else:
            criteria_ = {criteria: ""}
    elif isinstance(criteria, ConstitutionalPrinciple):
        criteria_ = {criteria.name: criteria.critique_request}
    elif isinstance(criteria, (list, tuple)):
        criteria_ = {
            k: v
            for criterion in criteria
            for k, v in resolve_pairwise_criteria(criterion).items()
        }
    else:
        if not criteria:
            raise ValueError(
                "Criteria cannot be empty. "
                "Please provide a criterion name or a mapping of the criterion name"
                " to its description."
            )
        criteria_ = dict(criteria)
    return criteria_


class PairwiseStringResultOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the PairwiseStringEvalChain.

    Attributes:
        _type (str): The type of the output parser.

    """

    @property
    def _type(self) -> str:
        """Return the type of the output parser.

        Returns:
            str: The type of the output parser.

        """
        return "pairwise_string_result"

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Dict: The parsed output.

        Raises:
            ValueError: If the verdict is invalid.

        """
        match = _FIND_DOUBLE_BRACKETS.search(text)

        if match:
            verdict = match.group(1)

        if not match or verdict not in {"A", "B", "C"}:
            raise ValueError(
                f"Invalid output: {text}. "
                "Output must contain a double bracketed string\
                 with the verdict 'A', 'B', or 'C'."
            )
        # C means the models are tied. Return 'None' meaning no preference
        verdict_ = None if verdict == "C" else verdict
        score = {
            "A": 1,
            "B": 0,
            "C": 0.5,
        }[verdict]
        return {
            "reasoning": text,
            "value": verdict_,
            "score": score,
        }


class PairwiseStringEvalChain(PairwiseStringEvaluator, LLMEvalChain, LLMChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    Example:
        >>> from langchain.chat_models import ChatOpenAI
        >>> from langchain.evaluation.comparison import PairwiseStringEvalChain
        >>> llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        >>> chain = PairwiseStringEvalChain.from_llm(llm=llm)
        >>> result = chain.evaluate_string_pairs(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     prediction_b = (
        ...        "The chemical formula for water is H2O, which means"
        ...        " there are two hydrogen atoms and one oxygen atom."
        ...     reference = "The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "value": "B",
        #    "comment": "Both responses accurately state"
        #       " that the chemical formula for water is H2O."
        #       " However, Response B provides additional information"
        # .     " by explaining what the formula means.\\n[[B]]"
        # }

    """

    output_key: str = "results"  #: :meta private:
    output_parser: BaseOutputParser = Field(
        default_factory=PairwiseStringResultOutputParser
    )

    class Config:
        """Configuration for the PairwiseStringEvalChain."""

        extra = Extra.ignore

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return False

    @property
    def requires_input(self) -> bool:
        """Return whether the chain requires an input.

        Returns:
            bool: True if the chain requires an input, False otherwise.

        """
        return True

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            str: The warning to show when reference is ignored.

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledPairwiseStringEvalChain"
            " (EvaluatorType.LABELED_PAIRWISE_STRING) instead."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[PromptTemplate] = None,
        criteria: Optional[Union[CRITERIA_TYPE, str]] = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """Initialize the PairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseChatModel): The LLM to use (GPT-4 recommended).
            prompt (PromptTemplate, optional): The prompt to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            PairwiseStringEvalChain: The initialized PairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        if not (
            isinstance(llm, (ChatOpenAI, AzureChatOpenAI))
            and llm.model_name.startswith("gpt-4")
        ):
            logger.warning(
                "This chain was only tested with GPT-4. \
Performance may be significantly worse with other models."
            )

        expected_input_vars = {"prediction", "prediction_b", "input", "criteria"}
        prompt_ = prompt or COMPARISON_TEMPLATE.partial(reference="")
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" if v else k for k, v in criteria_.items())
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ""
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)

    def _prepare_input(
        self,
        prediction: str,
        prediction_b: str,
        input: Optional[str],
        reference: Optional[str],
    ) -> dict:
        """Prepare the input for the chain.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference string, if any.

        Returns:
            dict: The prepared input for the chain.

        """
        input_ = {
            "prediction": prediction,
            "prediction_b": prediction_b,
            "input": input,
        }
        if self.requires_reference:
            input_["reference"] = reference
        return input_

    def _prepare_output(self, result: dict) -> dict:
        """Prepare the output."""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = self(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = await self.acall(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class LabeledPairwiseStringEvalChain(PairwiseStringEvalChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs,
     with labeled preferences.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[PromptTemplate] = None,
        criteria: Optional[Union[CRITERIA_TYPE, str]] = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """Initialize the LabeledPairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            criteria (Union[CRITERIA_TYPE, str], optional): The criteria to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            LabeledPairwiseStringEvalChain: The initialized LabeledPairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """  # noqa: E501
        expected_input_vars = {
            "prediction",
            "prediction_b",
            "input",
            "reference",
            "criteria",
        }
        prompt_ = prompt or COMPARISON_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria_.items())
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ""
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)
