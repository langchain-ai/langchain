"""Test the criteria eval chain."""


import pytest

from langchain.evaluation.criteria.eval_chain import (
    _SUPPORTED_CRITERIA,
    CriteriaEvalChain,
)
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_resolve_criteria() -> None:
    assert CriteriaEvalChain.resolve_criteria("helpfulness") == {
        "helpfulness": _SUPPORTED_CRITERIA["helpfulness"]
    }
    assert CriteriaEvalChain.resolve_criteria(["correctness"]) == {
        "correctness": _SUPPORTED_CRITERIA["correctness"]
    }


def test_criteria_eval_chain() -> None:
    chain = CriteriaEvalChain.from_llm(
        llm=FakeLLM(
            queries={"text": "The meaning of life\nY"}, sequential_responses=True
        ),
        criteria={"my criterion": "my criterion description"},
    )
    with pytest.warns(UserWarning, match=chain._skip_reference_warning):
        result = chain.evaluate_strings(
            prediction="my prediction", reference="my reference", input="my input"
        )
    assert result["reasoning"] == "The meaning of life"


def test_criteria_eval_chain_missing_reference() -> None:
    chain = CriteriaEvalChain.from_llm(
        llm=FakeLLM(
            queries={"text": "The meaning of life\nY"},
            sequential_responses=True,
        ),
        requires_reference=True,
        criteria={"my criterion": "my criterion description"},
    )
    with pytest.raises(ValueError):
        chain.evaluate_strings(prediction="my prediction", input="my input")


def test_implements_string_protocol() -> None:
    assert issubclass(CriteriaEvalChain, StringEvaluator)
