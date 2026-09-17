"""Test the `langchain_typesafe` public interface."""

from langchain_typesafe import __all__

EXPECTED_ALL = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "ClassificationResponse",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "TypeSafeAPIConnectionError",
    "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError",
    "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError",
    "TypeSafeClassifier",
    "TypeSafeError",
    "TypeSafeInternalServerError",
    "TypeSafeNotFoundError",
    "TypeSafePermissionDeniedError",
    "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError",
    "Usage",
    "__version__",
]


def test_all_imports() -> None:
    """Verify that `__all__` contains the intended public interface."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
