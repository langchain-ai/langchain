import pytest

from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper


@pytest.fixture
def api_client() -> RedditSearchAPIWrapper:
    return RedditSearchAPIWrapper()  # type: ignore[call-arg]


def assert_results_exists(results: list) -> None:
    if len(results) > 0:
        for result in results:
            assert "post_title" in result
            assert "post_author" in result
            assert "post_subreddit" in result
            assert "post_text" in result
            assert "post_url" in result
            assert "post_score" in result
            assert "post_category" in result
            assert "post_id" in result
    else:
        assert results == []


@pytest.mark.requires("praw")
def test_run_empty_query(api_client: RedditSearchAPIWrapper) -> None:
    """Test that run gives the correct answer with empty query."""
    search = api_client.run(
        query="", sort="relevance", time_filter="all", subreddit="all", limit=5
    )
    assert search == "Searching r/all did not find any posts:"


@pytest.mark.requires("praw")
def test_run_query(api_client: RedditSearchAPIWrapper) -> None:
    """Test that run gives the correct answer."""
    search = api_client.run(
        query="university",
        sort="relevance",
        time_filter="all",
        subreddit="funny",
        limit=5,
    )
    assert "University" in search


@pytest.mark.requires("praw")
def test_results_exists(api_client: RedditSearchAPIWrapper) -> None:
    """Test that results gives the correct output format."""
    search = api_client.results(
        query="What is the best programming language?",
        sort="relevance",
        time_filter="all",
        subreddit="all",
        limit=10,
    )
    assert_results_exists(search)


@pytest.mark.requires("praw")
def test_results_empty_query(api_client: RedditSearchAPIWrapper) -> None:
    """Test that results gives the correct output with empty query."""
    search = api_client.results(
        query="", sort="relevance", time_filter="all", subreddit="all", limit=10
    )
    assert search == []
