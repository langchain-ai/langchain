"""Configuration for run evaluators."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import RunEvaluator

from langchain.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance as EmbeddingDistanceEnum,
)
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.evaluation.string_distance.base import (
    StringDistance as StringDistanceEnum,
)


class EvalConfig(BaseModel):
    """Configuration for a given run evaluator.

    Parameters
    ----------
    evaluator_type : EvaluatorType
        The type of evaluator to use.

    Methods
    -------
    get_kwargs()
        Get the keyword arguments for the evaluator configuration.

    """

    evaluator_type: EvaluatorType

    def get_kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments for the load_evaluator call.

        Returns
        -------
        Dict[str, Any]
            The keyword arguments for the load_evaluator call.

        """
        kwargs = {}
        for field, val in self:
            if field == "evaluator_type":
                continue
            elif val is None:
                continue
            kwargs[field] = val
        return kwargs


class SingleKeyEvalConfig(EvalConfig):
    """Configuration for a run evaluator that only requires a single key."""

    reference_key: Optional[str] = None
    """The key in the dataset run to use as the reference string.
    If not provided, we will attempt to infer automatically."""
    prediction_key: Optional[str] = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: Optional[str] = None
    """The key from the traced run's inputs dictionary to use to represent the
    input. If not provided, it will be inferred automatically."""

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        # Filer out the keys that are not needed for the evaluator.
        for key in ["reference_key", "prediction_key", "input_key"]:
            kwargs.pop(key, None)
        return kwargs


class RunEvalConfig(BaseModel):
    """Configuration for a run evaluation.

    Parameters
    ----------
    evaluators : List[Union[EvaluatorType, EvalConfig]]
        Configurations for which evaluators to apply to the dataset run.
        Each can be the string of an :class:`EvaluatorType <langchain.evaluation.schema.EvaluatorType>`, such
        as EvaluatorType.QA, the evaluator type string ("qa"), or a configuration for a
        given evaluator (e.g., :class:`RunEvalConfig.QA <langchain.smith.evaluation.config.RunEvalConfig.QA>`).

    custom_evaluators : Optional[List[Union[RunEvaluator, StringEvaluator]]]
        Custom evaluators to apply to the dataset run.

    reference_key : Optional[str]
        The key in the dataset run to use as the reference string.
        If not provided, it will be inferred automatically.

    prediction_key : Optional[str]
        The key from the traced run's outputs dictionary to use to
        represent the prediction. If not provided, it will be inferred
        automatically.

    input_key : Optional[str]
        The key from the traced run's inputs dictionary to use to represent the
        input. If not provided, it will be inferred automatically.

    eval_llm : Optional[BaseLanguageModel]
        The language model to pass to any evaluators that use a language model.
    """  # noqa: E501

    evaluators: List[Union[EvaluatorType, str, EvalConfig]] = Field(
        default_factory=list
    )
    """Configurations for which evaluators to apply to the dataset run.
    Each can be the string of an
    :class:`EvaluatorType <langchain.evaluation.schema.EvaluatorType>`, such
    as `EvaluatorType.QA`, the evaluator type string ("qa"), or a configuration for a
    given evaluator
    (e.g., 
    :class:`RunEvalConfig.QA <langchain.smith.evaluation.config.RunEvalConfig.QA>`)."""  # noqa: E501
    custom_evaluators: Optional[List[Union[RunEvaluator, StringEvaluator]]] = None
    """Custom evaluators to apply to the dataset run."""
    reference_key: Optional[str] = None
    """The key in the dataset run to use as the reference string.
    If not provided, we will attempt to infer automatically."""
    prediction_key: Optional[str] = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: Optional[str] = None
    """The key from the traced run's inputs dictionary to use to represent the
    input. If not provided, it will be inferred automatically."""
    eval_llm: Optional[BaseLanguageModel] = None
    """The language model to pass to any evaluators that require one."""

    class Config:
        arbitrary_types_allowed = True

    class Criteria(SingleKeyEvalConfig):
        """Configuration for a reference-free criteria evaluator.

        Parameters
        ----------
        criteria : Optional[CRITERIA_TYPE]
            The criteria to evaluate.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.

        """

        criteria: Optional[CRITERIA_TYPE] = None
        llm: Optional[BaseLanguageModel] = None
        evaluator_type: EvaluatorType = EvaluatorType.CRITERIA

        def __init__(
            self, criteria: Optional[CRITERIA_TYPE] = None, **kwargs: Any
        ) -> None:
            super().__init__(criteria=criteria, **kwargs)

    class LabeledCriteria(SingleKeyEvalConfig):
        """Configuration for a labeled (with references) criteria evaluator.

        Parameters
        ----------
        criteria : Optional[CRITERIA_TYPE]
            The criteria to evaluate.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.
        """

        criteria: Optional[CRITERIA_TYPE] = None
        llm: Optional[BaseLanguageModel] = None
        evaluator_type: EvaluatorType = EvaluatorType.LABELED_CRITERIA

        def __init__(
            self, criteria: Optional[CRITERIA_TYPE] = None, **kwargs: Any
        ) -> None:
            super().__init__(criteria=criteria, **kwargs)

    class EmbeddingDistance(SingleKeyEvalConfig):
        """Configuration for an embedding distance evaluator.

        Parameters
        ----------
        embeddings : Optional[Embeddings]
            The embeddings to use for computing the distance.

        distance_metric : Optional[EmbeddingDistanceEnum]
            The distance metric to use for computing the distance.

        """

        evaluator_type: EvaluatorType = EvaluatorType.EMBEDDING_DISTANCE
        embeddings: Optional[Embeddings] = None
        distance_metric: Optional[EmbeddingDistanceEnum] = None

        class Config:
            arbitrary_types_allowed = True

    class StringDistance(SingleKeyEvalConfig):
        """Configuration for a string distance evaluator.

        Parameters
        ----------
        distance : Optional[StringDistanceEnum]
            The string distance metric to use.

        """

        evaluator_type: EvaluatorType = EvaluatorType.STRING_DISTANCE
        distance: Optional[StringDistanceEnum] = None
        """The string distance metric to use.
            damerau_levenshtein: The Damerau-Levenshtein distance.
            levenshtein: The Levenshtein distance.
            jaro: The Jaro distance.
            jaro_winkler: The Jaro-Winkler distance.
        """
        normalize_score: bool = True
        """Whether to normalize the distance to between 0 and 1.
        Applies only to the Levenshtein and Damerau-Levenshtein distances."""

    class QA(SingleKeyEvalConfig):
        """Configuration for a QA evaluator.

        Parameters
        ----------
        prompt : Optional[BasePromptTemplate]
            The prompt template to use for generating the question.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.
        """

        evaluator_type: EvaluatorType = EvaluatorType.QA
        llm: Optional[BaseLanguageModel] = None
        prompt: Optional[BasePromptTemplate] = None

    class ContextQA(SingleKeyEvalConfig):
        """Configuration for a context-based QA evaluator.

        Parameters
        ----------
        prompt : Optional[BasePromptTemplate]
            The prompt template to use for generating the question.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.

        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: Optional[BaseLanguageModel] = None
        prompt: Optional[BasePromptTemplate] = None

    class CoTQA(SingleKeyEvalConfig):
        """Configuration for a context-based QA evaluator.

        Parameters
        ----------
        prompt : Optional[BasePromptTemplate]
            The prompt template to use for generating the question.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.

        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: Optional[BaseLanguageModel] = None
        prompt: Optional[BasePromptTemplate] = None

    class JsonValidity(SingleKeyEvalConfig):
        """Configuration for a json validity evaluator.

        Parameters
        ----------
        """

        evaluator_type: EvaluatorType = EvaluatorType.JSON_VALIDITY

    class JsonEqualityEvaluator(EvalConfig):
        """Configuration for a json equality evaluator.

        Parameters
        ----------
        """

        evaluator_type: EvaluatorType = EvaluatorType.JSON_EQUALITY

    class ExactMatch(SingleKeyEvalConfig):
        """Configuration for an exact match string evaluator.

        Parameters
        ----------
        ignore_case : bool
            Whether to ignore case when comparing strings.
        ignore_punctuation : bool
            Whether to ignore punctuation when comparing strings.
        ignore_numbers : bool
            Whether to ignore numbers when comparing strings.
        """

        evaluator_type: EvaluatorType = EvaluatorType.STRING_DISTANCE
        ignore_case: bool = False
        ignore_punctuation: bool = False
        ignore_numbers: bool = False

    class RegexMatch(SingleKeyEvalConfig):
        """Configuration for a regex match string evaluator.

        Parameters
        ----------
        flags : int
            The flags to pass to the regex. Example: re.IGNORECASE.
        """

        evaluator_type: EvaluatorType = EvaluatorType.REGEX_MATCH
        flags: int = 0

    class ScoreString(SingleKeyEvalConfig):
        """Configuration for a score string evaluator.
        This is like the criteria evaluator but it is configured by
        default to return a score on the scale from 1-10.

        It is recommended to normalize these scores
        by setting `normalize_by` to 10.

        Parameters
        ----------
        criteria : Optional[CRITERIA_TYPE]
            The criteria to evaluate.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.
        normalize_by: Optional[int] = None
            If you want to normalize the score, the denominator to use.
            If not provided, the score will be between 1 and 10 (by default).
        prompt : Optional[BasePromptTemplate]

        """

        evaluator_type: EvaluatorType = EvaluatorType.SCORE_STRING
        criteria: Optional[CRITERIA_TYPE] = None
        llm: Optional[BaseLanguageModel] = None
        normalize_by: Optional[float] = None
        prompt: Optional[BasePromptTemplate] = None

        def __init__(
            self,
            criteria: Optional[CRITERIA_TYPE] = None,
            normalize_by: Optional[float] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(criteria=criteria, normalize_by=normalize_by, **kwargs)

    class LabeledScoreString(ScoreString):
        evaluator_type: EvaluatorType = EvaluatorType.LABELED_SCORE_STRING
