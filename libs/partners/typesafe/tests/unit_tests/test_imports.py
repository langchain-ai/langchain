"""Test the `langchain_typesafe` public interface."""

import typesafe_sdk as ts

from langchain_typesafe import Choice, Noul, Score, __all__

EXPECTED_ALL = [
    "Choice",
    "Noul",
    "Score",
    "State",
    "TypeSafeClassifier",
    "__version__",
]

# Owned by the SDK and deliberately not mirrored at this package's top level.
NOT_RE_EXPORTED = [
    "Answer",
    "ChoiceAnswer",
    "NoulAnswer",
    "NoulCriteria",
    "RetryPolicy",
    "ScoreAnswer",
    "SystemOneResponse",
    "TypeSafeAPIError",
    "TypeSafeRateLimitError",
    "Usage",
]


def test_all_imports() -> None:
    """Verify that `__all__` contains the intended public interface."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_sdk_types_are_not_mirrored() -> None:
    """SDK-owned types are not duplicated in this package's public interface."""
    assert not set(NOT_RE_EXPORTED) & set(__all__)
    # They remain importable from the SDK, which is where they are documented.
    for name in NOT_RE_EXPORTED:
        assert hasattr(ts, name)


def test_question_types_are_the_sdk_types() -> None:
    """Re-exported question types are the SDK's own, not redeclared copies."""
    assert Noul is ts.Noul
    assert Choice is ts.Choice
    assert Score is ts.Score
