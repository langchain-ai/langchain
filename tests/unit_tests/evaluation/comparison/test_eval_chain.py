"""Test the comparison chains."""


import re

import pytest

from langchain.evaluation.comparison.eval_chain import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
)
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_pairwise_string_comparison_chain() -> None:
    llm = FakeLLM(
        queries={
            "a": "The values are the same.\n[[C]]",
            "b": "A is clearly better than b.\n[[A]]",
            "c": "B is clearly better than a.\n[[B]]",
        },
        sequential_responses=True,
    )
    chain = PairwiseStringEvalChain.from_llm(llm=llm)
    res = chain.evaluate_string_pairs(
        prediction="I like pie.",
        prediction_b="I love pie.",
        input="What is your favorite food?",
    )
    assert res["value"] is None
    assert res["score"] == 0.5
    assert res["reasoning"] == "The values are the same."
    res = chain.evaluate_string_pairs(
        prediction="I like pie.",
        prediction_b="I like pie.",
        input="What is your favorite food?",
    )
    assert res["value"] == "A"
    assert res["score"] == 1
    with pytest.warns(UserWarning, match=re.escape(chain._skip_reference_warning)):
        res = chain.evaluate_string_pairs(
            prediction="I like pie.",
            prediction_b="I hate pie.",
            input="What is your favorite food?",
            reference="I enjoy pie.",
        )
    assert res["value"] == "B"
    assert res["score"] == 0


def test_labeled_pairwise_string_comparison_chain_missing_ref() -> None:
    llm = FakeLLM(
        queries={
            "a": "The values are the same.\n[[C]]",
            "b": "A is clearly better than b.\n[[A]]",
            "c": "B is clearly better than a.\n[[B]]",
        },
        sequential_responses=True,
    )
    chain = LabeledPairwiseStringEvalChain.from_llm(llm=llm)
    with pytest.raises(ValueError):
        chain.evaluate_string_pairs(
            prediction="I like pie.",
            prediction_b="I love pie.",
            input="What is your favorite food?",
        )
