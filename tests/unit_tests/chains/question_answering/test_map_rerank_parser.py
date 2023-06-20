"""Test map_rerank parser"""
from langchain.chains.question_answering import map_rerank_prompt

GOOD_SCORE = "foo bar answer.\nScore: 80"
SCORE_WITH_EXPLANATION = "foo bar answer.\nScore: 80 (fully answers the question, but could provide more detail on the specific error message)"

def test_good_score() -> None:
    parser = map_rerank_prompt.PROMPT.output_parser
    result = parser.parse(GOOD_SCORE)

    assert result["answer"] == "foo bar answer."

    score = int(result["score"])
    assert score == 80


def test_score_with_explanation() -> None:
    parser = map_rerank_prompt.PROMPT.output_parser
    result = parser.parse(SCORE_WITH_EXPLANATION)

    assert result["answer"] == "foo bar answer."

    score = int(result["score"])
    assert score == 80

