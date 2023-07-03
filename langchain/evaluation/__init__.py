"""Evaluation chains for grading LLM and Chain outputs.

This module contains off-the-shelf evaluation chains for grading the output of
LangChain primitives such as language models and chains.

To load an evaluator, you can use the :func:`load_evaluators <langchain.evaluation.loading.load_evaluators>` function with the
name of the evaluator to load.

To load one of the LangChain HuggingFace datasets, you can use the :func:`load_dataset <langchain.evaluation.loading.load_dataset>` function with the
name of the dataset to load.

Some common use cases for evaluation include:

- Grading the accuracy of a response against ground truth answers: :class:`QAEvalChain <langchain.evaluation.qa.eval_chain.QAEvalChain>`
- Comparing the output of two models: :class:`PairwiseStringEvalChain <langchain.evaluation.comparison.eval_chain.PairwiseStringEvalChain>`
- Judging the efficacy of an agent's tool usage: :class:`TrajectoryEvalChain <langchain.evaluation.agents.trajectory_eval_chain.TrajectoryEvalChain>`
- Checking whether an output complies with a set of criteria: :class:`CriteriaEvalChain <langchain.evaluation.criteria.eval_chain.CriteriaEvalChain>`

This module also contains low-level APIs for creating custom evaluators for
specific evaluation tasks. These include:

- :class:`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`: Evaluates an output string against a reference and/or input context.
- :class:`PairwiseStringEvaluator <langchain.evaluation.schema.PairwiseStringEvaluator>`: Evaluates two strings against each other.

"""  # noqa: E501
from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.evaluation.comparison.eval_chain import PairwiseStringEvalChain
from langchain.evaluation.criteria.eval_chain import CriteriaEvalChain
from langchain.evaluation.loading import load_dataset, load_evaluators
from langchain.evaluation.qa import ContextQAEvalChain, CotQAEvalChain, QAEvalChain
from langchain.evaluation.schema import (
    EvaluatorType,
    PairwiseStringEvaluator,
    StringEvaluator,
)

__all__ = [
    "EvaluatorType",
    "PairwiseStringEvalChain",
    "QAEvalChain",
    "CotQAEvalChain",
    "ContextQAEvalChain",
    "StringEvaluator",
    "PairwiseStringEvaluator",
    "TrajectoryEvalChain",
    "CriteriaEvalChain",
    "load_evaluators",
    "load_dataset",
]
