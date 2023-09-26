"""Scoring evaluators.

This module contains evaluators for scoring on a 1-10 the output of models,
be they LLMs, Chains, or otherwise. This can be based on a variety of
criteria and or a reference answer.

Example:
    >>> from langchain.chat_models import ChatOpenAI
    >>> from langchain.evaluation.comparison import PairwiseStringEvalChain
    >>> llm = ChatOpenAI(temperature=0)
    >>> chain = PairwiseStringEvalChain.from_llm(llm=llm)
    >>> result = chain.evaluate_string_pairs(
    ...     input = "What is the chemical formula for water?",
    ...     prediction = "H2O",
    ...     prediction_b = (
    ...        "The chemical formula for water is H2O, which means"
    ...        " there are two hydrogen atoms and one oxygen atom."
    ...     reference = "The chemical formula for water is H2O.",
    ... )
    >>> print(result["text"])
    # {
    #    "value": "B",
    #    "comment": "Both responses accurately state"
    #       " that the chemical formula for water is H2O."
    #       " However, Response B provides additional information"
    # .     " by explaining what the formula means.\\n[[B]]"
    # }
"""
from langchain.evaluation.comparison.eval_chain import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
)

__all__ = ["PairwiseStringEvalChain", "LabeledPairwiseStringEvalChain"]
