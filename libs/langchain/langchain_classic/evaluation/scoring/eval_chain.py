"""Base classes for scoring the output of a model on a scale of 1-10."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.criteria.eval_chain import (
    CRITERIA_TYPE,
    Criteria,
)
from langchain_classic.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain_classic.evaluation.scoring.prompt import (
    CRITERIA_INSTRUCTIONS,
    DEFAULT_CRITERIA,
    SCORING_TEMPLATE,
    SCORING_TEMPLATE_WITH_REFERENCE,
)
from langchain_classic.schema import RUN_KEY

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


def resolve_criteria(
    criteria: CRITERIA_TYPE | str | list[CRITERIA_TYPE] | None,
) -> dict:
    """Resolve the criteria for the pairwise evaluator.

    Args:
        criteria: The criteria to use.

    Returns:
        The resolved criteria.

    """
    if criteria is None:
        _default_criteria = [
            Criteria.HELPFULNESS,
            Criteria.RELEVANCE,
            Criteria.CORRECTNESS,
            Criteria.DEPTH,
        ]
        return {k.value: _SUPPORTED_CRITERIA[k] for k in _default_criteria}
    if isinstance(criteria, Criteria):
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
            for k, v in resolve_criteria(criterion).items()
        }
    else:
        if not criteria:
            msg = (
                "Criteria cannot be empty. "
                "Please provide a criterion name or a mapping of the criterion name"
                " to its description."
            )
            raise ValueError(msg)
        criteria_ = dict(criteria)
    return criteria_


class ScoreStringResultOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the ScoreStringEvalChain.

    Attributes:
        _type: The type of the output parser.

    """

    @property
    def _type(self) -> str:
        """Return the type of the output parser.

        Returns:
            The type of the output parser.

        """
        return "pairwise_string_result"

    def parse(self, text: str) -> dict[str, Any]:
        """Parse the output text.

        Args:
            text: The output text to parse.

        Returns:
            The parsed output.

        Raises:
            ValueError: If the verdict is invalid.

        """
        match = _FIND_DOUBLE_BRACKETS.search(text)

        if match:
            verdict = match.group(1)

        if not match or verdict not in [*list("123456789"), "10"]:
            msg = (
                f"Invalid output: {text}. "
                "Output must contain a double bracketed string\
                 with the verdict between 1 and 10."
            )
            raise ValueError(msg)

        return {
            "reasoning": text,
            "score": int(verdict),
        }


class ScoreStringEvalChain(StringEvaluator, LLMEvalChain, LLMChain):
    """A chain for scoring on a scale of 1-10 the output of a model.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_classic.evaluation.scoring import ScoreStringEvalChain
        >>> model = ChatOpenAI(temperature=0, model_name="gpt-4")
        >>> chain = ScoreStringEvalChain.from_llm(llm=model)
        >>> result = chain.evaluate_strings(
        ...     input="What is the chemical formula for water?",
        ...     prediction="H2O",
        ...     reference="The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "score": 8,
        #    "comment": "The response accurately states "
        #    "that the chemical formula for water is H2O."
        #    "However, it does not provide an explanation of what the formula means."
        # }

    """

    output_key: str = "results"
    output_parser: BaseOutputParser = Field(
        default_factory=ScoreStringResultOutputParser,
    )
    normalize_by: float | None = None
    """The value to normalize the score by, if specified."""
    criterion_name: str
    """The name of the criterion being evaluated."""

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            `True` if the chain requires a reference, `False` otherwise.

        """
        return False

    @property
    def requires_input(self) -> bool:
        """Return whether the chain requires an input.

        Returns:
            `True` if the chain requires an input, `False` otherwise.

        """
        return True

    @property
    def evaluation_name(self) -> str:
        """Get the name of the evaluation.

        Returns:
        -------
        str
            The name of the evaluation.
        """
        return f"score_string:{self.criterion_name}"

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            The warning to show when reference is ignored.

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledScoreStringEvalChain instead."
            " (EvaluatorType.LABELED_SCORE_STRING) instead."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        normalize_by: float | None = None,
        **kwargs: Any,
    ) -> ScoreStringEvalChain:
        """Initialize the ScoreStringEvalChain from an LLM.

        Args:
            llm: The LLM to use (GPT-4 recommended).
            prompt: The prompt to use.
            criteria: The criteria to use.
            normalize_by: The value to normalize the score by.
            **kwargs: Additional keyword arguments.

        Returns:
            The initialized ScoreStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        if not (hasattr(llm, "model_name") and not llm.model_name.startswith("gpt-4")):
            logger.warning(
                "This chain was only tested with GPT-4. \
Performance may be significantly worse with other models.",
            )

        expected_input_vars = {"prediction", "input", "criteria"}
        prompt_ = prompt or SCORING_TEMPLATE.partial(reference="")
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_criteria(criteria)
        criteria_str = "\n".join(
            f"{k}: {v}" if v else k for k, v in criteria_.items()
        ).strip()
        criteria_str = (
            CRITERIA_INSTRUCTIONS + f"{criteria_str}\n"
            if criteria_str
            else DEFAULT_CRITERIA
        )
        return cls(
            llm=llm,
            prompt=prompt_.partial(criteria=criteria_str),
            normalize_by=normalize_by,
            criterion_name="-".join(criteria_),
            **kwargs,
        )

    def _prepare_input(
        self,
        prediction: str,
        input_: str | None,
        reference: str | None,
    ) -> dict:
        """Prepare the input for the chain.

        Args:
            prediction: The output string from the first model.
            prediction_b: The output string from the second model.
            input_: The input or task string.
            reference: The reference string, if any.

        Returns:
            The prepared input for the chain.

        """
        input_dict = {
            "prediction": prediction,
            "input": input_,
        }
        if self.requires_reference:
            input_dict["reference"] = reference
        return input_dict

    def _prepare_output(self, result: dict) -> dict:
        """Prepare the output."""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        if "score" in parsed and self.normalize_by is not None:
            parsed["score"] = parsed["score"] / self.normalize_by
        return parsed

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Score the output string.

        Args:
            prediction: The output string from the first model.
            input: The input or task string.
            callbacks: The callbacks to use.
            tags: Optional tags to use.
            metadata: Optional metadata to use.
            include_run_info: Whether to include run information in the output.
            reference: The reference string, if any.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - reasoning: The reasoning for the preference.
                - score: A score between 1 and 10.

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = self(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously score the output string.

        Args:
            prediction: The output string from the first model.
            input: The input or task string.
            callbacks: The callbacks to use.
            tags: Optional tags to use.
            metadata: Optional metadata to use.
            include_run_info: Whether to include run information in the output.
            reference: The reference string, if any.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - reasoning: The reasoning for the preference.
                - score: A score between 1 and 10.

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = await self.acall(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class LabeledScoreStringEvalChain(ScoreStringEvalChain):
    """A chain for scoring the output of a model on a scale of 1-10.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            `True` if the chain requires a reference, `False` otherwise.

        """
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        normalize_by: float | None = None,
        **kwargs: Any,
    ) -> LabeledScoreStringEvalChain:
        """Initialize the LabeledScoreStringEvalChain from an LLM.

        Args:
            llm: The LLM to use.
            prompt: The prompt to use.
            criteria: The criteria to use.
            normalize_by: The value to normalize the score by.
            **kwargs: Additional keyword arguments.

        Returns:
            The initialized LabeledScoreStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        expected_input_vars = {
            "prediction",
            "input",
            "reference",
            "criteria",
        }
        prompt_ = prompt or SCORING_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria_.items()).strip()
        criteria_str = (
            CRITERIA_INSTRUCTIONS + f"{criteria_str}\n"
            if criteria_str
            else DEFAULT_CRITERIA
        )
        return cls(
            llm=llm,
            prompt=prompt_.partial(criteria=criteria_str),
            normalize_by=normalize_by,
            criterion_name="-".join(criteria_),
            **kwargs,
        )
