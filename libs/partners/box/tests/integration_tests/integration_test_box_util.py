import os

import pytest

from langchain_box.utilities import BoxAPIWrapper

box_config = {
    "BOX_FOLDER_ID": "199765563515",
    "BOX_FIRST_FILE": "1169674971571",
    "BOX_SECOND_FILE": "1169680553945",
    "BOX_ENTERPRISE_ID": "899905961",
    "BOX_USER_ID": "19498290761",
    "BOX_SEARCH_QUERY": "FIVE FEET AND RISING by Peter Sollett",
    "BOX_METADATA_QUERY": "total >: :value",
    "BOX_METADATA_PARAMS": '{ "value": "100"}',
    "BOX_METADATA_TEMPLATE": "InvoicePO",
    "BOX_AI_PROMPT": "list all the characters with a one sentence description",
}


@pytest.fixture
def box_api() -> BoxAPIWrapper:
    return BoxAPIWrapper(
        auth_type="ccg",
        box_client_id=os.environ["BOX_CLIENT_ID"],
        box_client_secret=os.environ["BOX_CLIENT_SECRET"],
        box_user_id=os.environ["BOX_USER_ID"],
    )  # type: ignore[call-arg]


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


def test_get_folder_items(box_api: BoxAPIWrapper) -> None:
    """Test that run gives the correct answer with empty query."""

    folder_contents = box_api.get_folder_items(box_config["BOX_FOLDER_ID"])
    assert "file" in folder_contents


# @pytest.mark.requires("praw")
# def test_run_query(api_client: RedditSearchAPIWrapper) -> None:
#     """Test that run gives the correct answer."""
#     search = api_client.run(
#         query="university",
#         sort="relevance",
#         time_filter="all",
#         subreddit="funny",
#         limit=5,
#     )
#     assert "University" in search


# @pytest.mark.requires("praw")
# def test_results_exists(api_client: RedditSearchAPIWrapper) -> None:
#     """Test that results gives the correct output format."""
#     search = api_client.results(
#         query="What is the best programming language?",
#         sort="relevance",
#         time_filter="all",
#         subreddit="all",
#         limit=10,
#     )
#     assert_results_exists(search)


# @pytest.mark.requires("praw")
# def test_results_empty_query(api_client: RedditSearchAPIWrapper) -> None:
#     """Test that results gives the correct output with empty query."""
#     search = api_client.results(
#         query="", sort="relevance", time_filter="all", subreddit="all", limit=10
#     )
#     assert search == []
