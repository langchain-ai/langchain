"""Test the scoring chains."""
import re

import pytest

from langchain.evaluation.scoring.eval_chain import (
    LabeledScoreStringEvalChain,
    ScoreStringEvalChain,
    ScoreStringResultOutputParser,
)
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_PairwiseStringResultOutputParser_parse() -> None:
    output_parser = ScoreStringResultOutputParser()
    text = """This answer is really good. 
Rating: [[10]]"""
    got = output_parser.parse(text)
    want = {
        "reasoning": text,
        "score": 10,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("score") == want["score"]

    text = """This answer is really good. 
Rating: 10"""
    with pytest.raises(ValueError):
        output_parser.parse(text)

    text = """This answer is really good. 
Rating: [[0]]"""
    # Not in range [1, 10]
    with pytest.raises(ValueError):
        output_parser.parse(text)


def test_pairwise_string_comparison_chain() -> None:
    llm = FakeLLM(
        queries={
            "a": "This is a rather good answer. Rating: [[9]]",
            "b": "This is a rather bad answer. Rating: [[1]]",
        },
        sequential_responses=True,
    )
    chain = ScoreStringEvalChain.from_llm(llm=llm)
    res = chain.evaluate_strings(
        prediction="I like pie.",
        input="What is your favorite food?",
    )
    assert res["score"] == 9
    assert res["reasoning"] == "This is a rather good answer. Rating: [[9]]"
    with pytest.warns(UserWarning, match=re.escape(chain._skip_reference_warning)):
        res = chain.evaluate_strings(
            prediction="I like pie.",
            input="What is your favorite food?",
            reference="I enjoy pie.",
        )
    assert res["score"] == 1
    assert res["reasoning"] == "This is a rather bad answer. Rating: [[1]]"


def test_labeled_pairwise_string_comparison_chain_missing_ref() -> None:
    llm = FakeLLM(
        queries={
            "a": "This is a rather good answer. Rating: [[9]]",
        },
        sequential_responses=True,
    )
    chain = LabeledScoreStringEvalChain.from_llm(llm=llm)
    with pytest.raises(ValueError):
        chain.evaluate_strings(
            prediction="I like pie.",
            input="What is your favorite food?",
        )
