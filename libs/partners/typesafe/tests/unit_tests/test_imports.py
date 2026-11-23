"""Test the `langchain_typesafe` public interface."""

from langchain_typesafe import __all__

EXPECTED_ALL = [
    "Answer",
    "Box",
    "BoxAnswer",
    "BoxCoordinates",
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "Content",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "TypeSafeClassifier",
    "__version__",
]


def test_all_imports() -> None:
    """Verify that `__all__` contains the intended public interface."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
