"""Configuration for run evaluators."""

from typing import Any, Dict, List, Optional, Union

from langsmith import RunEvaluator
from pydantic import BaseModel, Field

from langchain.embeddings.base import Embeddings
from langchain.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance as EmbeddingDistanceEnum,
)
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.evaluation.string_distance.base import (
    StringDistance as StringDistanceEnum,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate


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
            kwargs[field] = val
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

    evaluators: List[Union[EvaluatorType, EvalConfig]] = Field(default_factory=list)
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

    class Criteria(EvalConfig):
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

    class LabeledCriteria(EvalConfig):
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

    class EmbeddingDistance(EvalConfig):
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

    class StringDistance(EvalConfig):
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

    class QA(EvalConfig):
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

    class ContextQA(EvalConfig):
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

    class CoTQA(EvalConfig):
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

    # TODO: Trajectory
