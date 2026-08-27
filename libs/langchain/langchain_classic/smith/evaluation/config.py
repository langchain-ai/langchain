"""Configuration for run evaluators."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langsmith import RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain_classic.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain_classic.evaluation.embedding_distance.base import (
    EmbeddingDistance as EmbeddingDistanceEnum,
)
from langchain_classic.evaluation.schema import EvaluatorType, StringEvaluator
from langchain_classic.evaluation.string_distance.base import (
    StringDistance as StringDistanceEnum,
)

RUN_EVALUATOR_LIKE = Callable[
    [Run, Example | None],
    EvaluationResult | EvaluationResults | dict,
]
BATCH_EVALUATOR_LIKE = Callable[
    [Sequence[Run], Sequence[Example] | None],
    EvaluationResult | EvaluationResults | dict,
]


class EvalConfig(BaseModel):
    """Configuration for a given run evaluator.

    Attributes:
        evaluator_type: The type of evaluator to use.
    """

    evaluator_type: EvaluatorType

    def get_kwargs(self) -> dict[str, Any]:
        """Get the keyword arguments for the `load_evaluator` call.

        Returns:
            The keyword arguments for the `load_evaluator` call.
        """
        kwargs = {}
        for field, val in self:
            if field == "evaluator_type" or val is None:
                continue
            kwargs[field] = val
        return kwargs


class SingleKeyEvalConfig(EvalConfig):
    """Configuration for a run evaluator that only requires a single key."""

    reference_key: str | None = None
    """The key in the dataset run to use as the reference string.
    If not provided, we will attempt to infer automatically."""
    prediction_key: str | None = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: str | None = None
    """The key from the traced run's inputs dictionary to use to represent the
    input. If not provided, it will be inferred automatically."""

    @override
    def get_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_kwargs()
        # Filer out the keys that are not needed for the evaluator.
        for key in ["reference_key", "prediction_key", "input_key"]:
            kwargs.pop(key, None)
        return kwargs


CUSTOM_EVALUATOR_TYPE = RUN_EVALUATOR_LIKE | RunEvaluator | StringEvaluator
SINGLE_EVAL_CONFIG_TYPE = EvaluatorType | str | EvalConfig


class RunEvalConfig(BaseModel):
    """Configuration for a run evaluation."""

    evaluators: list[SINGLE_EVAL_CONFIG_TYPE | CUSTOM_EVALUATOR_TYPE] = Field(
        default_factory=list
    )
    """Configurations for which evaluators to apply to the dataset run.
    Each can be the string of an
    `EvaluatorType <langchain.evaluation.schema.EvaluatorType>`, such
    as `EvaluatorType.QA`, the evaluator type string ("qa"), or a configuration for a
    given evaluator
    (e.g.,
    `RunEvalConfig.QA <langchain.smith.evaluation.config.RunEvalConfig.QA>`)."""
    custom_evaluators: list[CUSTOM_EVALUATOR_TYPE] | None = None
    """Custom evaluators to apply to the dataset run."""
    batch_evaluators: list[BATCH_EVALUATOR_LIKE] | None = None
    """Evaluators that run on an aggregate/batch level.

    These generate one or more metrics that are assigned to the full test run.
    As a result, they are not associated with individual traces.
    """

    reference_key: str | None = None
    """The key in the dataset run to use as the reference string.
    If not provided, we will attempt to infer automatically."""
    prediction_key: str | None = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: str | None = None
    """The key from the traced run's inputs dictionary to use to represent the
    input. If not provided, it will be inferred automatically."""
    eval_llm: BaseLanguageModel | None = None
    """The language model to pass to any evaluators that require one."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    class Criteria(SingleKeyEvalConfig):
        """Configuration for a reference-free criteria evaluator.

        Attributes:
            criteria: The criteria to evaluate.
            llm: The language model to use for the evaluation chain.
        """

        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        evaluator_type: EvaluatorType = EvaluatorType.CRITERIA

    class LabeledCriteria(SingleKeyEvalConfig):
        """Configuration for a labeled (with references) criteria evaluator.

        Attributes:
            criteria: The criteria to evaluate.
            llm: The language model to use for the evaluation chain.
        """

        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        evaluator_type: EvaluatorType = EvaluatorType.LABELED_CRITERIA

    class EmbeddingDistance(SingleKeyEvalConfig):
        """Configuration for an embedding distance evaluator.

        Attributes:
            embeddings: The embeddings to use for computing the distance.
            distance_metric: The distance metric to use for computing the distance.
        """

        evaluator_type: EvaluatorType = EvaluatorType.EMBEDDING_DISTANCE
        embeddings: Embeddings | None = None
        distance_metric: EmbeddingDistanceEnum | None = None

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

    class StringDistance(SingleKeyEvalConfig):
        """Configuration for a string distance evaluator.

        Attributes:
            distance: The string distance metric to use (`damerau_levenshtein`,
                `levenshtein`, `jaro`, or `jaro_winkler`).
            normalize_score: Whether to normalize the distance to between 0 and 1.
                Applies only to the Levenshtein and Damerau-Levenshtein distances.
        """

        evaluator_type: EvaluatorType = EvaluatorType.STRING_DISTANCE
        distance: StringDistanceEnum | None = None
        normalize_score: bool = True

    class QA(SingleKeyEvalConfig):
        """Configuration for a QA evaluator.

        Attributes:
            prompt: The prompt template to use for generating the question.
            llm: The language model to use for the evaluation chain.
        """

        evaluator_type: EvaluatorType = EvaluatorType.QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class ContextQA(SingleKeyEvalConfig):
        """Configuration for a context-based QA evaluator.

        Attributes:
            prompt: The prompt template to use for generating the question.
            llm: The language model to use for the evaluation chain.
        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class CoTQA(SingleKeyEvalConfig):
        """Configuration for a context-based QA evaluator.

        Attributes:
            prompt: The prompt template to use for generating the question.
            llm: The language model to use for the evaluation chain.
        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class JsonValidity(SingleKeyEvalConfig):
        """Configuration for a json validity evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.JSON_VALIDITY

    class JsonEqualityEvaluator(EvalConfig):
        """Configuration for a json equality evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.JSON_EQUALITY

    class ExactMatch(SingleKeyEvalConfig):
        """Configuration for an exact match string evaluator.

        Attributes:
            ignore_case: Whether to ignore case when comparing strings.
            ignore_punctuation: Whether to ignore punctuation when comparing strings.
            ignore_numbers: Whether to ignore numbers when comparing strings.
        """

        evaluator_type: EvaluatorType = EvaluatorType.EXACT_MATCH
        ignore_case: bool = False
        ignore_punctuation: bool = False
        ignore_numbers: bool = False

    class RegexMatch(SingleKeyEvalConfig):
        """Configuration for a regex match string evaluator.

        Attributes:
            flags: The flags to pass to the regex. Example: `re.IGNORECASE`.
        """

        evaluator_type: EvaluatorType = EvaluatorType.REGEX_MATCH
        flags: int = 0

    class ScoreString(SingleKeyEvalConfig):
        """Configuration for a score string evaluator.

        This is like the criteria evaluator but it is configured by
        default to return a score on the scale from 1-10.

        It is recommended to normalize these scores
        by setting `normalize_by` to 10.

        Attributes:
            criteria: The criteria to evaluate.
            llm: The language model to use for the evaluation chain.
            normalize_by: If you want to normalize the score, the denominator to use.
                If not provided, the score will be between 1 and 10.
            prompt: The prompt template to use for evaluation.
        """

        evaluator_type: EvaluatorType = EvaluatorType.SCORE_STRING
        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        normalize_by: float | None = None
        prompt: BasePromptTemplate | None = None

    class LabeledScoreString(ScoreString):
        """Configuration for a labeled score string evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.LABELED_SCORE_STRING
