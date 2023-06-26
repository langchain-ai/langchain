"""Criteria or rubric based evaluators.

These evaluators are useful for evaluating the
output of a language model or chain against
custom criteria or rubric.

Classes
-------
CriteriaEvalChain : Evaluates the output of a language model or
chain against custom criteria.

Examples
--------
Using a pre-defined criterion:
>>> from langchain.llms import OpenAI
>>> from langchain.evaluation.criteria import CriteriaEvalChain

>>> llm = OpenAI()
>>> criteria = "conciseness"
>>> chain = CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)
>>> chain.evaluate_strings(
        prediction="The answer is 42.",
        reference="42",
        input="What is the answer to life, the universe, and everything?",
    )

Using a custom criterion:

>>> from langchain.llms import OpenAI
>>> from langchain.evaluation.criteria import CriteriaEvalChain

>>> llm = OpenAI()
>>> criteria = {
       "hallucination": (
            "Does this submission contain information"
            " not present in the input or reference?"
        ),
    }
>>> chain = CriteriaEvalChain.from_llm(
        llm=llm,
        criteria=criteria,
        requires_reference=True,
        )
"""

from langchain.evaluation.criteria.eval_chain import CriteriaEvalChain

__all__ = ["CriteriaEvalChain"]
