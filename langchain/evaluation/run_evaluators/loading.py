""""Loading helpers for run evaluators."""


from typing import Any, List, Optional, Sequence, Union

from langchainplus_sdk import RunEvaluator

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluators
from langchain.evaluation.run_evaluators.string_run_evaluator import (
    StringRunEvaluatorChain,
)
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.tools.base import Tool


def load_run_evaluators_for_model(
    evaluators: Sequence[EvaluatorType],
    model: Union[Chain, BaseLanguageModel, Tool],
    *,
    input_key: Optional[str] = None,
    prediction_key: Optional[str] = None,
    reference_key: Optional[str] = None,
    eval_llm: Optional[BaseLanguageModel] = None,
    **kwargs: Any,
) -> List[RunEvaluator]:
    """Load evaluators specified by a list of evaluator types.

    Parameters
    ----------
    evaluators : Sequence[EvaluatorType]
        The list of evaluator types to load.
    model : Union[Chain, BaseLanguageModel, Tool]
        The model to evaluate. Used to infer how to parse the run.
    input_key : Optional[str], a chain run's input key to map
        to the evaluator's input
    prediction_key : Optional[str], the key in the run's outputs to
        represent the Chain prediction
    reference_key : Optional[str], the key in the dataset example (row)
        outputs to represent the reference, or ground-truth label
    eval_llm : BaseLanguageModel, optional
        The language model to use for evaluation, if none is provided, a default
        ChatOpenAI gpt-4 model will be used.
    **kwargs : Any
        Additional keyword arguments to pass to all evaluators.

    Returns
    -------
    List[RunEvaluator]
        The loaded Run evaluators.
    """
    evaluators_ = load_evaluators(evaluators, llm=eval_llm, **kwargs)
    run_evaluators = []
    for evaluator in evaluators_:
        if isinstance(evaluator, StringEvaluator):
            run_evaluator = StringRunEvaluatorChain.from_model_and_evaluator(
                model,
                evaluator,
                input_key=input_key,
                prediction_key=prediction_key,
                reference_key=reference_key,
            )
        else:
            raise NotImplementedError(
                f"Run evaluator for {evaluator} is not implemented"
            )
        run_evaluators.append(run_evaluator)
    return run_evaluators
