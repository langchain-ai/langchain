"""Configuration for run evaluators."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.evaluation.criteria.eval_chain import Criteria
from langchain.evaluation.schema import EvaluatorType
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class RunEvaluatorConfig(BaseModel):
    """Configuration for a given run evaluator."""

    evaluator_type: EvaluatorType

    def get_loader_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for the load_evaluator function."""
        return {}


class RunEvaluationConfig(BaseModel):
    evaluator_configs: List[RunEvaluatorConfig]
    reference_key: Optional[str] = None
    prediction_key: Optional[str] = None
    input_key: Optional[str] = None
    eval_llm: Optional[BaseLanguageModel] = None


# TODO move to other files
class CriteriaEvaluatorConfig(RunEvaluatorConfig):
    """Configuration for a single criteria evaluator."""

    evaluator_type: EvaluatorType = EvaluatorType.CRITERIA
    criteria: Optional[Criteria] = None
    custom_criteria: Optional[Dict[str, str]] = None
    constitutional_principle: Optional[ConstitutionalPrinciple] = None

    @root_validator
    def validate_one_specified(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that only one of criteria, custom_criteria, or
        constitutional_principle is specified."""
        criteria = values.get("criteria")
        custom_criteria = values.get("custom_criteria")
        constitutional_principle = values.get("constitutional_principle")
        if (
            sum(
                [
                    criteria is not None,
                    custom_criteria is not None,
                    constitutional_principle is not None,
                ]
            )
            != 1
        ):
            raise ValueError(
                "Exactly one of criteria, custom_criteria, or constitutional_principle"
                " must be specified for the CriteriaEvalConfig."
            )
        return values

    def get_loader_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for the load_evaluator function."""
        criteria = (
            self.criteria or self.custom_criteria or self.constitutional_principle
        )
        return {"criteria": criteria}


class QAEvaluatorConfig(RunEvaluatorConfig):
    """Configuration for a single QA evaluator."""

    evaluator_type: EvaluatorType = EvaluatorType.QA
    prompt: Optional[PromptTemplate] = None


class ContextQAEvalChain(QAEvaluatorConfig):
    """Configuration for a single Context QA evaluator."""

    evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA


class COTQAEvaluatorConfig(QAEvaluatorConfig):
    """Configuration for a single Chain of thought QA evaluator."""

    evaluator_type: EvaluatorType = EvaluatorType.COT_QA
