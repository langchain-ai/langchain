from langchain_core.example_selectors import __all__

EXPECTED_ALL = [
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "SemanticSimilarityExampleSelector",
    "sorted_values",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
