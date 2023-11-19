
"""Tool for the Google search API."""

from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.pydantic_v1 import Field, BaseModel
from langchain.utilities.reddit_search import RedditSearchAPIWrapper

class RedditSearchSchema(BaseModel):
    query: str = Field(description='should be query string that post title should contain')
    sort: str = Field(description='should be sort method, which is one of: "relevance", "hot", "top", "new", or "comments".')
    time_filter: str = Field(description='should be time period to filter by, which is one of "all", "day", "hour", "month", "week", or "year"')
    subreddit: str = Field(description='should be name of subreddit, like "all" for r/all')

class RedditSearchRun(BaseTool):
    """Tool that queries the Google search API."""

    name: str = "google_search"
    description: str = (
        "A tool for searching Reddit. "
        "Useful for when you need to search for a post in a subreddit. "
        "Input should be search query, sort method, time filter, and subreddit name."
    )
    api_wrapper: RedditSearchAPIWrapper
    args_schema: Type[BaseModel] = RedditSearchSchema

    def _run(self,
        query: str,
        sort: str, 
        time_filter: str,
        subreddit: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(
            query=query, 
            sort=sort, 
            time_filter=time_filter, 
            subreddit=subreddit
        )

