"""Loading datasets and evaluators."""
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from langchain_core.language_models import BaseLanguageModel

from langchain.chains.base import Chain
from langchain.chat_models.openai import ChatOpenAI
from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.evaluation.comparison import PairwiseStringEvalChain
from langchain.evaluation.comparison.eval_chain import LabeledPairwiseStringEvalChain
from langchain.evaluation.criteria.eval_chain import (
    CriteriaEvalChain,
    LabeledCriteriaEvalChain,
)
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistanceEvalChain,
    PairwiseEmbeddingDistanceEvalChain,
)
from langchain.evaluation.exact_match.base import ExactMatchStringEvaluator
from langchain.evaluation.parsing.base import (
    JsonEqualityEvaluator,
    JsonValidityEvaluator,
)
from langchain.evaluation.parsing.json_distance import JsonEditDistanceEvaluator
from langchain.evaluation.parsing.json_schema import JsonSchemaEvaluator
from langchain.evaluation.qa import ContextQAEvalChain, CotQAEvalChain, QAEvalChain
from langchain.evaluation.regex_match.base import RegexMatchStringEvaluator
from langchain.evaluation.schema import EvaluatorType, LLMEvalChain, StringEvaluator
from langchain.evaluation.scoring.eval_chain import (
    LabeledScoreStringEvalChain,
    ScoreStringEvalChain,
)
from langchain.evaluation.string_distance.base import (
    PairwiseStringDistanceEvalChain,
    StringDistanceEvalChain,
)


def load_dataset(uri: str) -> List[Dict]:
    """Load a dataset from the `LangChainDatasets on HuggingFace <https://huggingface.co/LangChainDatasets>`_.

    Args:
        uri: The uri of the dataset to load.

    Returns:
        A list of dictionaries, each representing a row in the dataset.

    **Prerequisites**

    .. code-block:: shell

        pip install datasets

    Examples
    --------
    .. code-block:: python

        from langchain.evaluation import load_dataset
        ds = load_dataset("llm-math")
    """  # noqa: E501
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "load_dataset requires the `datasets` package."
            " Please install with `pip install datasets`"
        )

    dataset = load_dataset(f"LangChainDatasets/{uri}")
    return [d for d in dataset["train"]]


_EVALUATOR_MAP: Dict[
    EvaluatorType, Union[Type[LLMEvalChain], Type[Chain], Type[StringEvaluator]]
] = {
    EvaluatorType.QA: QAEvalChain,
    EvaluatorType.COT_QA: CotQAEvalChain,
    EvaluatorType.CONTEXT_QA: ContextQAEvalChain,
    EvaluatorType.PAIRWISE_STRING: PairwiseStringEvalChain,
    EvaluatorType.SCORE_STRING: ScoreStringEvalChain,
    EvaluatorType.LABELED_PAIRWISE_STRING: LabeledPairwiseStringEvalChain,
    EvaluatorType.LABELED_SCORE_STRING: LabeledScoreStringEvalChain,
    EvaluatorType.AGENT_TRAJECTORY: TrajectoryEvalChain,
    EvaluatorType.CRITERIA: CriteriaEvalChain,
    EvaluatorType.LABELED_CRITERIA: LabeledCriteriaEvalChain,
    EvaluatorType.STRING_DISTANCE: StringDistanceEvalChain,
    EvaluatorType.PAIRWISE_STRING_DISTANCE: PairwiseStringDistanceEvalChain,
    EvaluatorType.EMBEDDING_DISTANCE: EmbeddingDistanceEvalChain,
    EvaluatorType.PAIRWISE_EMBEDDING_DISTANCE: PairwiseEmbeddingDistanceEvalChain,
    EvaluatorType.JSON_VALIDITY: JsonValidityEvaluator,
    EvaluatorType.JSON_EQUALITY: JsonEqualityEvaluator,
    EvaluatorType.JSON_EDIT_DISTANCE: JsonEditDistanceEvaluator,
    EvaluatorType.JSON_SCHEMA_VALIDATION: JsonSchemaEvaluator,
    EvaluatorType.REGEX_MATCH: RegexMatchStringEvaluator,
    EvaluatorType.EXACT_MATCH: ExactMatchStringEvaluator,
}


def load_evaluator(
    evaluator: EvaluatorType,
    *,
    llm: Optional[BaseLanguageModel] = None,
    **kwargs: Any,
) -> Union[Chain, StringEvaluator]:
    """Load the requested evaluation chain specified by a string.

    Parameters
    ----------
    evaluator : EvaluatorType
        The type of evaluator to load.
    llm : BaseLanguageModel, optional
        The language model to use for evaluation, by default None
    **kwargs : Any
        Additional keyword arguments to pass to the evaluator.

    Returns
    -------
    Chain
        The loaded evaluation chain.

    Examples
    --------
    >>> from langchain.evaluation import load_evaluator, EvaluatorType
    >>> evaluator = load_evaluator(EvaluatorType.QA)
    """
    if evaluator not in _EVALUATOR_MAP:
        raise ValueError(
            f"Unknown evaluator type: {evaluator}"
            f"\nValid types are: {list(_EVALUATOR_MAP.keys())}"
        )
    evaluator_cls = _EVALUATOR_MAP[evaluator]
    if issubclass(evaluator_cls, LLMEvalChain):
        try:
            llm = llm or ChatOpenAI(
                model="gpt-4", model_kwargs={"seed": 42}, temperature=0
            )
        except Exception as e:
            raise ValueError(
                f"Evaluation with the {evaluator_cls} requires a "
                "language model to function."
                " Failed to create the default 'gpt-4' model."
                " Please manually provide an evaluation LLM"
                " or check your openai credentials."
            ) from e
        return evaluator_cls.from_llm(llm=llm, **kwargs)
    else:
        return evaluator_cls(**kwargs)


def load_evaluators(
    evaluators: Sequence[EvaluatorType],
    *,
    llm: Optional[BaseLanguageModel] = None,
    config: Optional[dict] = None,
    **kwargs: Any,
) -> List[Union[Chain, StringEvaluator]]:
    """Load evaluators specified by a list of evaluator types.

    Parameters
    ----------
    evaluators : Sequence[EvaluatorType]
        The list of evaluator types to load.
    llm : BaseLanguageModel, optional
        The language model to use for evaluation, if none is provided, a default
        ChatOpenAI gpt-4 model will be used.
    config : dict, optional
        A dictionary mapping evaluator types to additional keyword arguments,
        by default None
    **kwargs : Any
        Additional keyword arguments to pass to all evaluators.

    Returns
    -------
    List[Chain]
        The loaded evaluators.

    Examples
    --------
    >>> from langchain.evaluation import load_evaluators, EvaluatorType
    >>> evaluators = [EvaluatorType.QA, EvaluatorType.CRITERIA]
    >>> loaded_evaluators = load_evaluators(evaluators, criteria="helpfulness")
    """
    loaded = []
    for evaluator in evaluators:
        _kwargs = config.get(evaluator, {}) if config else {}
        loaded.append(load_evaluator(evaluator, llm=llm, **{**kwargs, **_kwargs}))
    return loaded
