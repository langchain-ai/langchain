"""Test the `langchain_typesafe` public interface."""

import typesafe_sdk as ts

from langchain_typesafe import (
    Choice,
    Noul,
    NoulCriteria,
    RetryPolicy,
    Score,
    __all__,
)

EXPECTED_ALL = [
    "Choice",
    "Noul",
    "NoulCriteria",
    "RetryPolicy",
    "Score",
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
    "__version__",
]

# Answer and response types stay in the SDK rather than being mirrored here.
NOT_RE_EXPORTED = [
    "Answer",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
]


def test_all_imports() -> None:
    """Verify that `__all__` contains the intended public interface."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_sdk_types_are_not_mirrored() -> None:
    """SDK-owned answer and response types are not duplicated in this package."""
    assert not set(NOT_RE_EXPORTED) & set(__all__)
    # They remain importable from the SDK, which is where they are documented.
    for name in NOT_RE_EXPORTED:
        assert hasattr(ts, name)


def test_question_types_are_the_sdk_types() -> None:
    """Re-exported question types are the SDK's own, not redeclared copies."""
    assert Noul is ts.Noul
    assert Choice is ts.Choice
    assert Score is ts.Score
    assert NoulCriteria is ts.NoulCriteria
    assert RetryPolicy is ts.RetryPolicy
