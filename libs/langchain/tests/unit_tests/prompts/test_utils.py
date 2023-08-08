"""Test functionality related to prompt utils."""
from langchain.prompts.example_selector.semantic_similarity import sorted_values


def test_sorted_vals() -> None:
    """Test sorted values from dictionary."""
    test_dict = {"key2": "val2", "key1": "val1"}
    expected_response = ["val1", "val2"]
    assert sorted_values(test_dict) == expected_response
