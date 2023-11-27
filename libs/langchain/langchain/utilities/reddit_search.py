"""Wrapper for the Reddit API"""

from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env


class RedditSearchAPIWrapper(BaseModel):
    """Wrapper for Reddit API

    To use, set the environment variables ``REDDIT_CLIENT_ID``,
    ``REDDIT_CLIENT_SECRET``, ``REDDIT_USER_AGENT`` to set the client ID,
    client secret, and user agent, respectively, as given by Reddit's API.
    Alternatively, all three can be supplied as named parameters in the
    constructor: ``reddit_client_id``, ``reddit_client_secret``, and
    ``reddit_user_agent``, respectively.

    Example:
        .. code-block:: python

            from langchain.utilities import RedditSearchAPIWrapper
            reddit_search = RedditSearchAPIWrapper()
    """

    reddit_client: Any

    # Values required to access Reddit API via praw
    reddit_client_id: Optional[str]
    reddit_client_secret: Optional[str]
    reddit_user_agent: Optional[str]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API ID, secret and user agent exists in environment
        and check that praw module is present.
        """
        reddit_client_id = get_from_dict_or_env(
            values, "reddit_client_id", "REDDIT_CLIENT_ID"
        )
        values["reddit_client_id"] = reddit_client_id

        reddit_client_secret = get_from_dict_or_env(
            values, "reddit_client_secret", "REDDIT_CLIENT_SECRET"
        )
        values["reddit_client_secret"] = reddit_client_secret

        reddit_user_agent = get_from_dict_or_env(
            values, "reddit_user_agent", "REDDIT_USER_AGENT"
        )
        values["reddit_user_agent"] = reddit_user_agent

        try:
            import praw
        except ImportError:
            raise ImportError(
                "praw package not found, please install it with pip install praw"
            )

        reddit_client = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )
        values["reddit_client"] = reddit_client

        return values

    def run(
        self, query: str, sort: str, time_filter: str, subreddit: str, limit: int
    ) -> str:
        """Search Reddit and return posts as a single string."""
        results: List[Dict] = self.results(
            query=query,
            sort=sort,
            time_filter=time_filter,
            subreddit=subreddit,
            limit=limit,
        )
        if len(results) > 0:
            output: List[str] = [f"Searching r/{subreddit} found {len(results)} posts:"]
            for r in results:
                category = "N/A" if r["post_category"] is None else r["post_category"]
                p = f"Post Title: '{r['post_title']}'\n\
                    User: {r['post_author']}\n\
                    Subreddit: {r['post_subreddit']}:\n\
                    Text body: {r['post_text']}\n\
                    Post URL: {r['post_url']}\n\
                    Post Category: {category}.\n\
                    Score: {r['post_score']}\n"
                output.append(p)
            return "\n".join(output)
        else:
            return f"Searching r/{subreddit} did not find any posts:"

    def results(
        self, query: str, sort: str, time_filter: str, subreddit: str, limit: int
    ) -> List[Dict]:
        """Use praw to search Reddit and return a list of dictionaries,
        one for each post.
        """
        subredditObject = self.reddit_client.subreddit(subreddit)
        search_results = subredditObject.search(
            query=query, sort=sort, time_filter=time_filter, limit=limit
        )
        search_results = [r for r in search_results]
        results_object = []
        for submission in search_results:
            results_object.append(
                {
                    "post_subreddit": submission.subreddit_name_prefixed,
                    "post_category": submission.category,
                    "post_title": submission.title,
                    "post_text": submission.selftext,
                    "post_score": submission.score,
                    "post_id": submission.id,
                    "post_url": submission.url,
                    "post_author": submission.author,
                }
            )
        return results_object
