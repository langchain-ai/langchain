"""Criteria or rubric based evaluators.

These evaluators are useful for evaluating the
output of a language model or chain against
specified criteria or rubric.

Classes
-------
CriteriaEvalChain : Evaluates the output of a language model or
chain against specified criteria.

Examples:
--------
Using a predefined criterion:
>>> from langchain_openai import OpenAI
>>> from langchain_classic.evaluation.criteria import CriteriaEvalChain

>>> model = OpenAI()
>>> criteria = "conciseness"
>>> chain = CriteriaEvalChain.from_llm(llm=model, criteria=criteria)
>>> chain.evaluate_strings(
        prediction="The answer is 42.",
        reference="42",
        input="What is the answer to life, the universe, and everything?",
    )

Using a custom criterion:

>>> from langchain_openai import OpenAI
>>> from langchain_classic.evaluation.criteria import LabeledCriteriaEvalChain

>>> model = OpenAI()
>>> criteria = {
       "hallucination": (
            "Does this submission contain information"
            " not present in the input or reference?"
        ),
    }
>>> chain = LabeledCriteriaEvalChain.from_llm(
        llm=model,
        criteria=criteria,
        )
>>> chain.evaluate_strings(
        prediction="The answer to life is 42.",
        reference="It's commonly known that the answer to life is 42.",
        input="Please summarize the following: The answer to life, the universe, and everything is unknowable.",
    )
"""  # noqa: E501

from langchain_classic.evaluation.criteria.eval_chain import (
    Criteria,
    CriteriaEvalChain,
    LabeledCriteriaEvalChain,
)

__all__ = ["Criteria", "CriteriaEvalChain", "LabeledCriteriaEvalChain"]
