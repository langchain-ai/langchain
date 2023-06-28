"""Functionality relating to evaluation.

This module contains off-the-shelf evaluation chains for
grading the output of LangChain primitives such as LLMs and Chains.

Some common use cases for evaluation include:

- Grading accuracy of a response against ground truth answers: QAEvalChain
- Comparing the output of two models: PairwiseStringEvalChain
- Judging the efficacy of an agent's tool usage: TrajectoryEvalChain
- Checking whether an output complies with a set of criteria: CriteriaEvalChain

This module also contains low level APIs for making more evaluators for your
custom evaluation task. These include:
- StringEvaluator: Evaluates an output string against a reference and/or
    with input context.
- PairwiseStringEvaluator: Evaluates two strings against each other.
"""

from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.evaluation.comparison import PairwiseStringEvalChain
from langchain.evaluation.criteria.eval_chain import CriteriaEvalChain
from langchain.evaluation.qa import ContextQAEvalChain, CotQAEvalChain, QAEvalChain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator

__all__ = [
    "PairwiseStringEvalChain",
    "QAEvalChain",
    "CotQAEvalChain",
    "ContextQAEvalChain",
    "StringEvaluator",
    "PairwiseStringEvaluator",
    "TrajectoryEvalChain",
    "CriteriaEvalChain",
]
