"""Integration test for Wolfram Alpha API Wrapper."""

from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = WolframAlphaAPIWrapper()  # type: ignore[call-arg]
    output = search.run("what is 2x+18=x+5?")
    assert "x = -13" in output
