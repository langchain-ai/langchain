"""Tool for the Reddit search API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper


class RedditSearchSchema(BaseModel):
    """Input for Reddit search."""

    query: str = Field(
        description="should be query string that post title should \
        contain, or '*' if anything is allowed."
    )
    sort: str = Field(
        description='should be sort method, which is one of: "relevance" \
        , "hot", "top", "new", or "comments".'
    )
    time_filter: str = Field(
        description='should be time period to filter by, which is \
        one of "all", "day", "hour", "month", "week", or "year"'
    )
    subreddit: str = Field(
        description='should be name of subreddit, like "all" for \
        r/all'
    )
    limit: str = Field(
        description="a positive integer indicating the maximum number \
        of results to return"
    )
    include_comment_forest: bool = Field(
        default=False,
        description="A boolean indicating whether to include the comment "
        "forest in the results. Defaults to False, which avoids large data pulls.",
    )


class RedditSearchRun(BaseTool):  # type: ignore[override, override]
    """Tool that queries for posts on a subreddit,
    optionally fetching full comment forest."""

    name: str = "reddit_search"
    description: str = (
        "A tool that searches for posts on Reddit. "
        "Optionally, it can fetch the entire comment forest. "
        "Useful when you need to know post information on a subreddit."
    )
    api_wrapper: RedditSearchAPIWrapper = Field(default_factory=RedditSearchAPIWrapper)  # type: ignore[arg-type]
    # Add a constructor param to allow (or disallow) comment forest fetching at all.
    allow_comment_forest: bool = Field(
        default=False,
        description="If False, all calls will ignore `include_comment_forest=True`. "
        "Set to True to allow the agent/model to fetch full comment trees.",
    )
    args_schema: Type[BaseModel] = RedditSearchSchema

    def _run(
        self,
        query: str,
        sort: str,
        time_filter: str,
        subreddit: str,
        limit: str,
        include_comment_forest: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # If we do not allow comment forest fetching, force it to False
        if not self.allow_comment_forest:
            include_comment_forest = False

        return self.api_wrapper.run(
            query=query,
            sort=sort,
            time_filter=time_filter,
            subreddit=subreddit,
            limit=int(limit),
            include_comment_forest=include_comment_forest,
        )
