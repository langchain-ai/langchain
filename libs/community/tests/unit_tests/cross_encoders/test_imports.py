from langchain_community.cross_encoders import __all__, _module_lookup

EXPECTED_ALL = [
    "BaseCrossEncoder",
    "FakeCrossEncoder",
    "HuggingFaceCrossEncoder",
    "SagemakerEndpointCrossEncoder",
]


def test_all_imports() -> None:
    """Test that __all__ is correctly set."""
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
