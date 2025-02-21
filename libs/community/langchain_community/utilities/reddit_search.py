"""Wrapper for the Reddit API"""

from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator


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

            from langchain_community.utilities import RedditSearchAPIWrapper
            reddit_search = RedditSearchAPIWrapper()
    """

    reddit_client: Any

    # Values required to access Reddit API via praw
    reddit_client_id: Optional[str]
    reddit_client_secret: Optional[str]
    reddit_user_agent: Optional[str]

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
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
        self,
        query: str,
        sort: str,
        time_filter: str,
        subreddit: str,
        limit: int,
        include_comment_forest: bool = False,
    ) -> str:
        """Search Reddit and return posts as a single string."""
        results: List[Dict] = self.results(
            query=query,
            sort=sort,
            time_filter=time_filter,
            subreddit=subreddit,
            limit=limit,
            include_comment_forest=include_comment_forest,
        )
        if len(results) == 0:
            return f"Searching r/{subreddit} did not find any posts."

        output: List[str] = [f"Searching r/{subreddit} found {len(results)} posts:"]
        for r in results:
            category = "N/A" if r["post_category"] is None else r["post_category"]
            p = (
                f"Post Title: '{r['post_title']}'\n"
                f"  Created: {r['post_created']}\n"
                f"  Post ID: {r['post_id']}\n"
                f"  User: {r['post_author']}\n"
                f"  Subreddit: {r['post_subreddit']}\n"
                f"  Text body: {r['post_text']}\n"
                f"  Post URL: {r['post_url']}\n"
                f"  Post Category: {category}\n"
                f"  Score: {r['post_score']}\n"
                f"  Upvote Ratio: {r['post_upvote_ratio']}\n"
            )
            output.append(p)

            # If requested, format and display the entire comment forest
            if include_comment_forest and r.get("post_comments"):
                output.append("  Comments (entire comment tree):")
                comment_str = self._format_comment_forest(r["post_comments"], indent=4)
                output.append(comment_str)
            output.append("===")

        return "\n".join(output)

    def _parse_comment_forest(self, comment_forest: Any) -> List[Dict[str, Any]]:
        """Recursively traverse the entire comment forest and return a list
        of dictionaries with comment info, including nested replies.
        """
        comments_data = []
        for comment in comment_forest:
            # Sometimes comment could be 'MoreComments' object
            if hasattr(comment, "body"):
                comment_info = {
                    "id": comment.id,
                    "body": comment.body,
                    "score": comment.score,
                    "ups": comment.ups,
                    "author": str(comment.author),
                    "created_utc": comment.created_utc,
                    # Recursively parse any replies (the nested forest)
                    "replies": self._parse_comment_forest(comment.replies),
                }
                comments_data.append(comment_info)
        return comments_data

    def _format_comment_forest(
        self, comments: List[Dict[str, Any]], indent: int = 0
    ) -> str:
        """Recursively build a readable string of the entire comment forest."""
        lines = []
        for idx, c in enumerate(comments, start=1):
            prefix = " " * indent + f"{idx}. "
            # Replace newlines in comment body to avoid messing up formatting
            body_single_line = c["body"].replace("\n", " ")
            lines.append(
                f"{prefix}[id: {c['id']}, score: {c['score']}, ups: {c['ups']}] "
                f"(by {c['author']}) {body_single_line}"
            )
            # If there are replies, recurse deeper
            if c["replies"]:
                replies_str = self._format_comment_forest(
                    c["replies"], indent=indent + 4
                )
                lines.append(replies_str)
        return "\n".join(lines)

    def results(
        self,
        query: str,
        sort: str,
        time_filter: str,
        subreddit: str,
        limit: int,
        include_comment_forest: bool = False,
    ) -> List[Dict]:
        """Use praw to search Reddit and return a list of dictionaries,
        one for each post. If include_comments is True, fetch the entire
        nested comment forest.
        """
        subreddit_obj = self.reddit_client.subreddit(subreddit)
        search_results = subreddit_obj.search(
            query=query, sort=sort, time_filter=time_filter, limit=limit
        )

        results_object = []
        for submission in search_results:
            post_data = {
                "post_subreddit": submission.subreddit_name_prefixed,
                "post_category": submission.category,
                "post_title": submission.title,
                "post_text": submission.selftext,
                "post_score": submission.score,
                "post_id": submission.id,
                "post_url": submission.url,
                "post_author": str(submission.author),
                "post_created": submission.created_utc,
                "post_upvote_ratio": submission.upvote_ratio,
                "post_ups": submission.ups,
            }

            # If include_comments, get the entire nested comment tree
            if include_comment_forest:
                submission.comments.replace_more(
                    limit=None
                )  # fetch all nested comments
                post_data["post_comments"] = self._parse_comment_forest(
                    submission.comments
                )

            results_object.append(post_data)

        return results_object
