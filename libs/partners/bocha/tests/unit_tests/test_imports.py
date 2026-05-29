"""Test that all classes can be imported successfully."""

from langchain_bocha import (
    BochaSearchResults,
    BochaSearchRun,
    ChatBocha,
)

EXPECTED_ALL = [
    "BochaSearchRun",
    "BochaSearchResults",
    "ChatBocha",
]


def test_all_imports() -> None:
    """Verify that all classes are present in the __all__ checklist."""
    assert sorted(EXPECTED_ALL) == sorted(
        [
            "BochaSearchRun",
            "BochaSearchResults",
            "ChatBocha",
        ]
    )


def test_instantiation() -> None:
    """Verify that classes can be instantiated successfully with fake keys."""
    run_tool = BochaSearchRun(bocha_api_key="fake")  # type: ignore[arg-type]
    assert run_tool is not None

    results_tool = BochaSearchResults(bocha_api_key="fake")  # type: ignore[arg-type]
    assert results_tool is not None

    model = ChatBocha(api_key="fake")  # type: ignore[arg-type]
    assert model is not None
