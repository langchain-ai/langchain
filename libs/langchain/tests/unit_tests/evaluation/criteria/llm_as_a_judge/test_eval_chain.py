"""Tests for "LLM-as-a-judge" comparison method."""
import pytest

from langchain.evaluation.comparison.llm_as_a_judge.eval_chain import (
    LLMAsAJudgePairwiseOutputParser,
)
from langchain.evaluation.comparison.llm_as_a_judge.prompt import (
    COMPARISON_TEMPLATE,
)


def test_comparsion_template() -> None:
    assert set(COMPARISON_TEMPLATE.input_variables) == {
        "input",
        "prediction",
        "prediction_b",
    }


@pytest.mark.parametrize(
    "text, verdict",
    [
        ("[Some reasoning] \n\nFinal Verdict: [[A]]", "Win"),
        ("[Some reasoning] Therefore [[B]] is better than [[A]].", "Loss"),
        ("[Some reasoning] \n\nFinal Verdict: [[C]]", "Tie"),
    ],
)
def test_llm_as_a_judge_pairwise_output_parser(text: str, verdict: str) -> None:
    parser = LLMAsAJudgePairwiseOutputParser()
    result = parser.parse(text)
    assert set(result.keys()) == {"reasoning", "verdict"}
    assert result["reasoning"] == text
    assert result["verdict"] == verdict
