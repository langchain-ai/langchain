"""Test the `langchain_typesafe` public interface."""

import typesafe_sdk as ts

from langchain_typesafe import __all__

EXPECTED_ALL = [
    "AutoModeMiddleware",
    "ModelChoice",
    "ModelRouterMiddleware",
    "Skill",
    "SkillSource",
    "SkillsMiddleware",
    "__version__",
]

# Owned by the SDK and deliberately not mirrored here. Question, answer, response,
# retry, and exception types are imported from `typesafe_sdk` directly.
NOT_RE_EXPORTED = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "RetryPolicy",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "TypeSafeAPIError",
    "TypeSafeClient",
    "TypeSafeRateLimitError",
    "Usage",
]


def test_all_imports() -> None:
    """Verify that `__all__` contains the intended public interface."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_sdk_types_are_not_mirrored() -> None:
    """This package exports middleware, not a copy of the SDK's surface."""
    assert not set(NOT_RE_EXPORTED) & set(__all__)
    for name in NOT_RE_EXPORTED:
        assert hasattr(ts, name)
