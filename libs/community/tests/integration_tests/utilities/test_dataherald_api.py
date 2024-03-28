"""Integration test for Dataherald API Wrapper."""
from langchain_community.utilities.dataherald import DataheraldAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = DataheraldAPIWrapper()
    output = search.run(
        "How many employees are in the company?", "65fb766367dd22c99ce1a12d"
    )
    assert "Answer: SELECT \n    COUNT(*) FROM \n    employees" in output
