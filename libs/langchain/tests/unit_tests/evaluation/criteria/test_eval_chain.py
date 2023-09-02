"""Test the criteria eval chain."""


import pytest

from langchain.evaluation.criteria.eval_chain import (
    _SUPPORTED_CRITERIA,
    Criteria,
    CriteriaEvalChain,
    CriteriaResultOutputParser,
    LabeledCriteriaEvalChain,
)
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_resolve_criteria_str() -> None:
    # type: ignore
    assert CriteriaEvalChain.resolve_criteria("helpfulness") == {
        "helpfulness": _SUPPORTED_CRITERIA[Criteria.HELPFULNESS]
    }
    assert CriteriaEvalChain.resolve_criteria("correctness") == {
        "correctness": _SUPPORTED_CRITERIA[Criteria.CORRECTNESS]
    }


def test_CriteriaResultOutputParser_parse() -> None:
    output_parser = CriteriaResultOutputParser()
    text = """Here is my step-by-step reasoning for the given criteria:
The criterion is: "Do you like cake?" I like cake.
Y"""
    got = output_parser.parse(text)
    want = {
        "reasoning": """Here is my step-by-step reasoning for the given criteria:
The criterion is: "Do you like cake?" I like cake.""",
        "value": "Y",
        "score": 1,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]

    text = "Y"
    got = output_parser.parse(text)
    want = {
        "reasoning": "",
        "value": "Y",
        "score": 1,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]


@pytest.mark.parametrize("criterion", list(Criteria))
def test_resolve_criteria_enum(criterion: Criteria) -> None:
    assert CriteriaEvalChain.resolve_criteria(criterion) == {
        criterion.value: _SUPPORTED_CRITERIA[criterion]
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
    chain = LabeledCriteriaEvalChain.from_llm(
        llm=FakeLLM(
            queries={"text": "The meaning of life\nY"},
            sequential_responses=True,
        ),
        criteria={"my criterion": "my criterion description"},
    )
    with pytest.raises(ValueError):
        chain.evaluate_strings(prediction="my prediction", input="my input")


def test_implements_string_protocol() -> None:
    assert issubclass(CriteriaEvalChain, StringEvaluator)
