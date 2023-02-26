"""Integration test for ArXiv Search API Wrapper."""
from langchain.utilities.arxiv_search import ArXivSearchAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = ArXivSearchAPIWrapper(max_results=1)
    output = search.run("quantum")
    assert "Stephen Blaha" in output


def test_no_result_call() -> None:
    """Test that call gives no result."""
    search = ArXivSearchAPIWrapper()
    output = search.run(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    print(type(output))
    assert "No good Google Search Result was found" == output
