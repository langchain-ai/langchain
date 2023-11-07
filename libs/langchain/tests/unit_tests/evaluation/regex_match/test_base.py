import re

import pytest

from langchain.evaluation import RegexMatchStringEvaluator


@pytest.fixture
def regex_match_string_evaluator() -> RegexMatchStringEvaluator:
    """Create a RegexMatchStringEvaluator with default configuration."""
    return RegexMatchStringEvaluator()


@pytest.fixture
def regex_match_string_evaluator_ignore_case() -> RegexMatchStringEvaluator:
    """Create a RegexMatchStringEvaluator with IGNORECASE flag."""
    return RegexMatchStringEvaluator(flags=re.IGNORECASE)


def test_default_regex_matching(
    regex_match_string_evaluator: RegexMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "^Mindy.*CTO$"
    result = regex_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 1.0

    reference = "^Mike.*CEO$"
    result = regex_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 0.0


def test_regex_matching_with_ignore_case(
    regex_match_string_evaluator_ignore_case: RegexMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "^mindy.*cto$"
    result = regex_match_string_evaluator_ignore_case.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 1.0
