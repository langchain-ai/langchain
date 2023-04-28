"""Integration test for Wikipedia API Wrapper."""
from langchain.utilities import WikipediaAPIWrapper


def test_call() -> None:
    """Test that WikipediaAPIWrapper returns correct answer"""

    wikipedia = WikipediaAPIWrapper()
    output = wikipedia.run("HUNTER X HUNTER")
    assert "Yoshihiro Togashi" in output


def test_no_result_call() -> None:
    """Test that call gives no result."""
    wikipedia = WikipediaAPIWrapper()
    output = wikipedia.run(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    assert "No good Wikipedia Search Result was found" == output
