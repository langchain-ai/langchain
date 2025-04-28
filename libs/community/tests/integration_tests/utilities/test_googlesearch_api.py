"""Integration test for Google Search API Wrapper."""

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSearchAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


def test_no_result_call() -> None:
    """Test that call gives no result."""
    search = GoogleSearchAPIWrapper()
    output = search.run(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    print(type(output))  # noqa: T201
    assert "No good Google Search Result was found" == output


def test_result_with_params_call() -> None:
    """Test that call gives the correct answer with extra params."""
    search = GoogleSearchAPIWrapper()
    output = search.results(
        query="What was Obama's first name?",
        num_results=5,
        search_params={"cr": "us", "safe": "active"},
    )
    assert len(output)
