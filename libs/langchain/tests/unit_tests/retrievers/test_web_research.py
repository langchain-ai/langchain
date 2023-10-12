from typing import List

import pytest

from langchain.retrievers.web_research import QuestionListOutputParser


@pytest.mark.parametrize(
    "text,expected",
    (
        (
            "1. Line one.\n",
            ["1. Line one.\n"],
        ),
        (
            "1. Line one.",
            ["1. Line one."],
        ),
        (
            "1. Line one.\n2. Line two.\n",
            ["1. Line one.\n", "2. Line two.\n"],
        ),
        (
            "1. Line one.\n2. Line two.",
            ["1. Line one.\n", "2. Line two."],
        ),
        (
            "1. Line one.\n2. Line two.\n3. Line three.",
            ["1. Line one.\n", "2. Line two.\n", "3. Line three."],
        ),
    ),
)
def test_list_output_parser(text: str, expected: List[str]) -> None:
    parser = QuestionListOutputParser()
    result = parser.parse(text)
    assert result.lines == expected
