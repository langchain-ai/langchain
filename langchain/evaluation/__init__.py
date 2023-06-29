"""Evaluation chains for grading LLM and Chain outputs.

This module contains off-the-shelf evaluation chains for grading the output of
LangChain primitives such as language models and chains.

Some common use cases for evaluation include:

- Grading the accuracy of a response against ground truth answers: QAEvalChain
- Comparing the output of two models: PairwiseStringEvalChain
- Judging the efficacy of an agent's tool usage: TrajectoryEvalChain
- Checking whether an output complies with a set of criteria: CriteriaEvalChain

This module also contains low-level APIs for creating custom evaluators for
specific evaluation tasks. These include:
- StringEvaluator: Evaluates an output string against a reference and/or input context.
- PairwiseStringEvaluator: Evaluates two strings against each other.


For loading evaluators and LangChain's HuggingFace datasets, you can use the
load_evaluators and load_dataset functions, respectively.
"""
from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.evaluation.comparison import PairwiseStringEvalChain
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
