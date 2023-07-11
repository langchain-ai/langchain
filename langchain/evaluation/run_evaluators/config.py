"""Configuration for run evaluators."""
from typing import Any, Dict, List, Optional, Union

from langsmith import RunEvaluator
from pydantic import BaseModel

from langchain.embeddings.base import Embeddings
from langchain.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain.evaluation.embedding_distance.base import EmbeddingDistance
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.evaluation.string_distance.base import StringDistance
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate


class EvalConfig(BaseModel):
    """Configuration for a given run evaluator."""

    evaluator_type: EvaluatorType
    """The type of evaluator to use."""

    def get_kwargs(self) -> Dict[str, Any]:
        return self.dict(exclude={"evaluator_type"}, exclude_none=True)


class RunEvalConfig(BaseModel):
    """Configuration for a run evaluation."""

    evaluators: List[Union[EvaluatorType, EvalConfig]]
    custom_evaluators: Optional[List[Union[RunEvaluator, StringEvaluator]]] = None
    reference_key: Optional[str] = None
    prediction_key: Optional[str] = None
    input_key: Optional[str] = None
    eval_llm: Optional[BaseLanguageModel] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, evaluators: List[Union[EvaluatorType, EvalConfig]], **kwargs: Any
    ):
        super().__init__(evaluators=evaluators, **kwargs)

    class Criteria(EvalConfig):
        """Configuration for a criteria evaluator."""

        criteria: Optional[CRITERIA_TYPE] = None
        evaluator_type: EvaluatorType = EvaluatorType.CRITERIA

        def __init__(
            self, criteria: Optional[CRITERIA_TYPE] = None, **kwargs: Any
        ) -> None:
            super().__init__(criteria=criteria, **kwargs)

    # TODO: LabeledCriteria when that's split out

    class EmbeddingDistance(EvalConfig):
        """Configuration for an embedding distance evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.EMBEDDING_DISTANCE
        embeddings: Optional[Embeddings] = None
        distance_metric: Optional[EmbeddingDistance] = None

        class Config:
            arbitrary_types_allowed = True

    class StringDistance(EvalConfig):
        """Configuration for a string distance evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.STRING_DISTANCE
        distance: Optional[StringDistance] = None

    class QA(EvalConfig):
        """Configuration for a QA evaluator."""

        evaluator_type: EvaluatorType = EvaluatorType.QA
        prompt: Optional[BasePromptTemplate] = None

    class ContextQA(EvalConfig):
        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        prompt: Optional[BasePromptTemplate] = None

    class CoTQA(EvalConfig):
        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        prompt: Optional[BasePromptTemplate] = None

    # TODO: Trajectory
