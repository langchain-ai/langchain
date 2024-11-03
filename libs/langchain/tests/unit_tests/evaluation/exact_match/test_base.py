import pytest

from langchain.evaluation import ExactMatchStringEvaluator


@pytest.fixture
def exact_match_string_evaluator() -> ExactMatchStringEvaluator:
    """Create an ExactMatchStringEvaluator with default configuration."""
    return ExactMatchStringEvaluator()


@pytest.fixture
def exact_match_string_evaluator_ignore_case() -> ExactMatchStringEvaluator:
    """Create an ExactMatchStringEvaluator with ignore_case set to True."""
    return ExactMatchStringEvaluator(ignore_case=True)


def test_default_exact_matching(
    exact_match_string_evaluator: ExactMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "Mindy is the CTO"
    result = exact_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 1.0

    reference = "Mindy is the CEO"
    result = exact_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 0.0


def test_exact_matching_with_ignore_case(
    exact_match_string_evaluator_ignore_case: ExactMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "mindy is the cto"
    result = exact_match_string_evaluator_ignore_case.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 1.0

    reference = "mindy is the CEO"
    result = exact_match_string_evaluator_ignore_case.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 0.0
