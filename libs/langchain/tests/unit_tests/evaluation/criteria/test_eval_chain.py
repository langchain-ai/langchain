"""Test the criteria eval chain."""

import pytest

from langchain_classic.evaluation.criteria.eval_chain import (
    _SUPPORTED_CRITERIA,
    Criteria,
    CriteriaEvalChain,
    CriteriaResultOutputParser,
    LabeledCriteriaEvalChain,
)
from langchain_classic.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_resolve_criteria_str() -> None:
    assert CriteriaEvalChain.resolve_criteria("helpfulness") == {
        "helpfulness": _SUPPORTED_CRITERIA[Criteria.HELPFULNESS],
    }
    assert CriteriaEvalChain.resolve_criteria("correctness") == {
        "correctness": _SUPPORTED_CRITERIA[Criteria.CORRECTNESS],
    }


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Y", {"reasoning": "", "value": "Y", "score": 1}),
        (
            "Here is my step-by-step reasoning for the given criteria:\n"
            'The criterion is: "Do you like cake?" I like cake.\n'
            "Y",
            {
                "reasoning": "Here is my step-by-step reasoning for the given criteria:"
                '\nThe criterion is: "Do you like cake?" I like cake.',
                "value": "Y",
                "score": 1,
            },
        ),
        (
            " NThe submission N is correct, accurate, and factual. It accurately"
            " identifies the specific effects of knowledge and interest on"
            " these factors. Therefore, the submission Y meets the criteria. Y",
            {
                "reasoning": "NThe submission N is correct, accurate, and factual. It"
                " accurately identifies the specific effects of knowledge and interest"
                " on these factors. Therefore, the submission Y meets the criteria.",
                "value": "Y",
                "score": 1,
            },
        ),
    ],
)
def test_criteria_result_output_parser_parse(text: str, want: dict) -> None:
    output_parser = CriteriaResultOutputParser()
    got = output_parser.parse(text)
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # Reasoning says "does not meet" but verdict is Y -> should flip to N
        (
            "The submission does not set the departure date at all, so it is"
            " incomplete. The submission does not meet the criteria of"
            " correctness because it fails to perform all required actions.\n\nY",
            {"reasoning_contains": "does not meet", "value": "N", "score": 0},
        ),
        # Reasoning says "doesn't meet" but verdict is Y -> should flip to N
        (
            "The answer is factually wrong. It doesn't meet the criteria.\n\nY",
            {"reasoning_contains": "doesn't meet", "value": "N", "score": 0},
        ),
        # Reasoning says "does not fulfill" but verdict is Y -> should flip to N
        (
            "The submission does not fulfill the objective as specified.\n\nY",
            {"reasoning_contains": "does not fulfill", "value": "N", "score": 0},
        ),
        # Reasoning says "meets the criteria" but verdict is N -> should flip to Y
        (
            "The submission is accurate and complete. It meets the criteria.\n\nN",
            {"reasoning_contains": "meets the criteria", "value": "Y", "score": 1},
        ),
        # No contradiction — verdict should stay as-is
        (
            "The submission is accurate and complete. It meets the criteria.\n\nY",
            {"reasoning_contains": "meets the criteria", "value": "Y", "score": 1},
        ),
        # Both positive and negative signals — ambiguous, don't flip
        (
            "The submission meets the criteria for relevance but does not meet"
            " the criteria for correctness.\n\nY",
            {
                "reasoning_contains": "does not meet",
                "value": "Y",
                "score": 1,
            },
        ),
    ],
)
def test_criteria_result_output_parser_consistency_check(
    text: str, want: dict
) -> None:
    """Test that the parser corrects verdict-reasoning inconsistencies."""
    output_parser = CriteriaResultOutputParser()
    got = output_parser.parse(text)
    assert want["reasoning_contains"] in got["reasoning"]
    assert got["value"] == want["value"]
    assert got["score"] == want["score"]


@pytest.mark.parametrize("criterion", list(Criteria))
def test_resolve_criteria_enum(criterion: Criteria) -> None:
    assert CriteriaEvalChain.resolve_criteria(criterion) == {
        criterion.value: _SUPPORTED_CRITERIA[criterion],
    }


def test_criteria_eval_chain() -> None:
    chain = CriteriaEvalChain.from_llm(
        llm=FakeLLM(
            queries={"text": "The meaning of life\nY"},
            sequential_responses=True,
        ),
        criteria={"my criterion": "my criterion description"},
    )
    with pytest.warns(UserWarning, match=chain._skip_reference_warning):
        result = chain.evaluate_strings(
            prediction="my prediction",
            reference="my reference",
            input="my input",
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
    with pytest.raises(
        ValueError, match="LabeledCriteriaEvalChain requires a reference string"
    ):
        chain.evaluate_strings(prediction="my prediction", input="my input")


def test_implements_string_protocol() -> None:
    assert issubclass(CriteriaEvalChain, StringEvaluator)
