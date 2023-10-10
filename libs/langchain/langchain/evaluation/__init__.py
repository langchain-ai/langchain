"""**Evaluation** chains for grading LLM and Chain outputs.

This module contains off-the-shelf evaluation chains for grading the output of
LangChain primitives such as language models and chains.

**Loading an evaluator**

To load an evaluator, you can use the :func:`load_evaluators <langchain.evaluation.loading.load_evaluators>` or
:func:`load_evaluator <langchain.evaluation.loading.load_evaluator>` functions with the
names of the evaluators to load.

.. code-block:: python

    from langchain.evaluation import load_evaluator

    evaluator = load_evaluator("qa")
    evaluator.evaluate_strings(
        prediction="We sold more than 40,000 units last week",
        input="How many units did we sell last week?",
        reference="We sold 32,378 units",
    )

The evaluator must be one of :class:`EvaluatorType <langchain.evaluation.schema.EvaluatorType>`.

**Datasets**

To load one of the LangChain HuggingFace datasets, you can use the :func:`load_dataset <langchain.evaluation.loading.load_dataset>` function with the
name of the dataset to load.

.. code-block:: python

        from langchain.evaluation import load_dataset
        ds = load_dataset("llm-math")

**Some common use cases for evaluation include:**

- Grading the accuracy of a response against ground truth answers: :class:`QAEvalChain <langchain.evaluation.qa.eval_chain.QAEvalChain>`
- Comparing the output of two models: :class:`PairwiseStringEvalChain <langchain.evaluation.comparison.eval_chain.PairwiseStringEvalChain>` or :class:`LabeledPairwiseStringEvalChain <langchain.evaluation.comparison.eval_chain.LabeledPairwiseStringEvalChain>` when there is additionally a reference label.
- Judging the efficacy of an agent's tool usage: :class:`TrajectoryEvalChain <langchain.evaluation.agents.trajectory_eval_chain.TrajectoryEvalChain>`
- Checking whether an output complies with a set of criteria: :class:`CriteriaEvalChain <langchain.evaluation.criteria.eval_chain.CriteriaEvalChain>` or :class:`LabeledCriteriaEvalChain <langchain.evaluation.criteria.eval_chain.LabeledCriteriaEvalChain>` when there is additionally a reference label.
- Computing semantic difference between a prediction and reference: :class:`EmbeddingDistanceEvalChain <langchain.evaluation.embedding_distance.base.EmbeddingDistanceEvalChain>` or between two predictions: :class:`PairwiseEmbeddingDistanceEvalChain <langchain.evaluation.embedding_distance.base.PairwiseEmbeddingDistanceEvalChain>` 
- Measuring the string distance between a prediction and reference :class:`StringDistanceEvalChain <langchain.evaluation.string_distance.base.StringDistanceEvalChain>` or between two predictions :class:`PairwiseStringDistanceEvalChain <langchain.evaluation.string_distance.base.PairwiseStringDistanceEvalChain>`

**Low-level API**

These evaluators implement one of the following interfaces:

- :class:`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`: Evaluate a prediction string against a reference label and/or input context.
- :class:`PairwiseStringEvaluator <langchain.evaluation.schema.PairwiseStringEvaluator>`: Evaluate two prediction strings against each other. Useful for scoring preferences, measuring similarity between two chain or llm agents, or comparing outputs on similar inputs.
- :class:`AgentTrajectoryEvaluator <langchain.evaluation.schema.AgentTrajectoryEvaluator>` Evaluate the full sequence of actions taken by an agent.

These interfaces enable easier composability and usage within a higher level evaluation framework.

"""  # noqa: E501
from langchain.evaluation.agents import TrajectoryEvalChain
from langchain.evaluation.comparison import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
)
from langchain.evaluation.criteria import (
    Criteria,
    CriteriaEvalChain,
    LabeledCriteriaEvalChain,
)
from langchain.evaluation.embedding_distance import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
    PairwiseEmbeddingDistanceEvalChain,
)
from langchain.evaluation.exact_match.base import ExactMatchStringEvaluator
from langchain.evaluation.loading import load_dataset, load_evaluator, load_evaluators
from langchain.evaluation.qa import ContextQAEvalChain, CotQAEvalChain, QAEvalChain
from langchain.evaluation.regex_match.base import RegexMatchStringEvaluator
from langchain.evaluation.schema import (
    AgentTrajectoryEvaluator,
    EvaluatorType,
    PairwiseStringEvaluator,
    StringEvaluator,
)
from langchain.evaluation.scoring import (
    LabeledScoreStringEvalChain,
    ScoreStringEvalChain,
)
from langchain.evaluation.string_distance import (
    PairwiseStringDistanceEvalChain,
    StringDistance,
    StringDistanceEvalChain,
)

__all__ = [
    "EvaluatorType",
    "ExactMatchStringEvaluator",
    "RegexMatchStringEvaluator",
    "PairwiseStringEvalChain",
    "LabeledPairwiseStringEvalChain",
    "QAEvalChain",
    "CotQAEvalChain",
    "ContextQAEvalChain",
    "StringEvaluator",
    "PairwiseStringEvaluator",
    "TrajectoryEvalChain",
    "CriteriaEvalChain",
    "Criteria",
    "EmbeddingDistance",
    "EmbeddingDistanceEvalChain",
    "PairwiseEmbeddingDistanceEvalChain",
    "StringDistance",
    "StringDistanceEvalChain",
    "PairwiseStringDistanceEvalChain",
    "LabeledCriteriaEvalChain",
    "load_evaluators",
    "load_evaluator",
    "load_dataset",
    "AgentTrajectoryEvaluator",
    "ScoreStringEvalChain",
    "LabeledScoreStringEvalChain",
]
