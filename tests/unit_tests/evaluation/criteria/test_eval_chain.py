"""Test the criteria eval chain."""


from langchain.evaluation.criteria.eval_chain import (
    HELPFULNESS_CRITERION,
    CriteriaEvalChain,
)
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_resolve_criteria() -> None:
    assert CriteriaEvalChain.resolve_criteria("helpfulness") == HELPFULNESS_CRITERION
    assert CriteriaEvalChain.resolve_criteria(["helpfulness"]) == HELPFULNESS_CRITERION


def test_criteria_eval_chain() -> None:
    chain = CriteriaEvalChain.from_criteria(
        llm=FakeLLM(queries={"foo": "The meaning of life"}, sequential_responses=True),
        criteria={"my criterion": "my criterion description"},
    )
    result = chain.evaluate_strings(
        prediction="my prediction", reference="my reference", input="my input"
    )
    assert result["text"] == "The meaning of life"


def test_implements_string_protocol() -> None:
    assert isinstance(CriteriaEvalChain, StringEvaluator)
