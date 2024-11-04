"""Integration test for Dataherald API Wrapper."""

from langchain_community.utilities.dataherald import DataheraldAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = DataheraldAPIWrapper(db_connection_id="65fb766367dd22c99ce1a12d")  # type: ignore[call-arg]
    output = search.run("How many employees are in the company?")
    assert "Answer: SELECT \n    COUNT(*) FROM \n    employees" in output
