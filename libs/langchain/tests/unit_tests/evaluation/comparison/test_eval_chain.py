"""Test the comparison chains."""


import re

import pytest

from langchain.evaluation.comparison.eval_chain import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
    PairwiseStringResultOutputParser,
    resolve_pairwise_criteria,
)
from langchain.evaluation.criteria.eval_chain import Criteria
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.parametrize("criterion", list(Criteria))
def test_resolve_criteria_enum(criterion: Criteria) -> None:
    val = resolve_pairwise_criteria(criterion)
    assert isinstance(val, dict)
    assert next(iter(val)) == criterion.value


def test_resolve_criteria_list_enum() -> None:
    val = resolve_pairwise_criteria(list(Criteria))
    assert isinstance(val, dict)
    assert set(val.keys()) == set(c.value for c in list(Criteria))


def test_PairwiseStringResultOutputParser_parse() -> None:
    output_parser = PairwiseStringResultOutputParser()
    text = """I like pie better than cake.
[[A]]"""
    got = output_parser.parse(text)
    want = {
        "reasoning": "I like pie better than cake.",
        "value": "A",
        "score": 1,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]

    text = """I like cake better than pie.
[[B]]"""
    got = output_parser.parse(text)
    want = {
        "reasoning": "I like cake better than pie.",
        "value": "B",
        "score": 0,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]

    text = """I like cake and pie.
[[C]]"""
    got = output_parser.parse(text)
    want = {
        "reasoning": "I like cake and pie.",
        "value": None,
        "score": 0.5,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]


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
