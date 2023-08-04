"""Test map_rerank parser"""
import pytest

from langchain.chains.question_answering.map_rerank_prompt import output_parser

GOOD_SCORE = "foo bar answer.\nScore: 80"
SCORE_WITH_EXPLANATION = "foo bar answer.\nScore: 80 (fully answers the question, but could provide more detail on the specific error message)"  # noqa: E501


@pytest.mark.parametrize("answer", (GOOD_SCORE, SCORE_WITH_EXPLANATION))
def test_parse_scores(answer: str) -> None:
    result = output_parser.parse(answer)

    assert result["answer"] == "foo bar answer."

    score = int(result["score"])
    assert score == 80
