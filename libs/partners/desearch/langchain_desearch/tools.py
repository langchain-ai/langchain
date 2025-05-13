import os
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, model_validator, root_validator
from desearch_py import Desearch
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema


class DesearchToolInput(BaseModel):
    prompt: str = Field(description="The search prompt or query.")
    # tool: str = Field(description="The specific tool to use (e.g., 'desearch_ai', 'desearch_web').")
    tool: List[
        Literal[
            "web", "hackernews", "reddit", "wikipedia", "youtube", "twitter", "arxiv"
        ]
    ] = Field(description="List of tools to use. Must include at least one tool.")
    model: str = Field(
        default="NOVA",
        description="The model to use for the search. Value should 'NOVA', 'ORBIT' or 'HORIZON'",
    )
    date_filter: Optional[str] = Field(
        default=None, description="Date filter for the search."
    )
    streaming: Optional[bool] = Field(
        default=False, description="Whether to stream results."
    )

    @model_validator(mode="after")
    def check_tool_non_empty(cls, values):
        tools = values.get("tool")
        if not tools:
            raise ValueError("The 'tool' field must contain at least one valid tool.")
        return values


class DesearchTool(BaseTool):
    name: str = "desearch_tool"
    description: str = (
        "Performs different Desearch API searches like AI search, Web links search, and Twitter posts search."
    )
    args_schema: ArgsSchema = DesearchToolInput
    return_direct: bool = True

    def _run(
        self,
        prompt: str,
        tool: List[str],
        model: str,
        date_filter: Optional[str] = None,
        streaming: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        if model not in ["NOVA", "ORBIT", "HORIZON"]:
            raise ValueError("Model should be 'NOVA', 'ORBIT' or 'HORIZON'")

        try:
            return desearch.ai_search(
                prompt=prompt,
                tools=tool,
                model=model,
                date_filter=date_filter,
                streaming=streaming,
            )
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        prompt: str,
        tool: str,
        model: str,
        date_filter: Optional[str] = None,
        streaming: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            prompt,
            tool,
            model,
            date_filter,
            streaming,
            run_manager=run_manager.get_sync(),
        )


class BasicWebSearchToolInput(BaseModel):
    query: str = Field(description="The search query.")
    num: int = Field(default=10, description="Number of results to return.")
    start: int = Field(
        default=1, description="The starting index for the search results."
    )


class BasicWebSearchTool(BaseTool):
    name: str = "basic_web_search_tool"
    description: str = "Performs a basic web search using Desearch."
    args_schema: ArgsSchema = BasicWebSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.basic_web_search(query=query, num=num, start=start)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(query, num, start, run_manager=run_manager.get_sync())


class BasicTwitterSearchToolInput(BaseModel):
    query: str = Field(description="The Twitter search query.")
    sort: str = Field(default="Top", description="Sort order for the results.")
    count: int = Field(default=10, description="Number of results to return.")


class BasicTwitterSearchTool(BaseTool):
    name: str = "basic_twitter_search_tool"
    description: str = "Performs a basic Twitter search using Desearch."
    args_schema: ArgsSchema = BasicTwitterSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        query: str,
        sort: str = "Top",
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.basic_twitter_search(query=query, sort=sort, count=count)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        query: str,
        sort: str = "Top",
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(query, sort, count, run_manager=run_manager.get_sync())


class FetchTweetsByUrlsToolInput(BaseModel):
    urls: list = Field(description="A list of Twitter URLs to fetch tweets from.")


class FetchTweetsByUrlsTool(BaseTool):
    name: str = "fetch_tweets_by_urls_tool"
    description: str = "Fetch tweets from the provided URLs."
    args_schema: ArgsSchema = FetchTweetsByUrlsToolInput
    return_direct: bool = True

    def _run(
        self,
        urls: list,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.twitter_by_urls(urls=urls)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        urls: list,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(urls, run_manager=run_manager.get_sync())


class FetchTweetsByIdToolInput(BaseModel):
    id: str = Field(description="The ID of the tweet to fetch.")


class FetchTweetsByIdTool(BaseTool):
    name: str = "fetch_tweets_by_id_tool"
    description: str = "Fetch a tweet by its ID."
    args_schema: ArgsSchema = FetchTweetsByIdToolInput
    return_direct: bool = True

    def _run(
        self,
        id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.twitter_by_id(id=id)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(id, run_manager=run_manager.get_sync())


class FetchLatestTweetsToolInput(BaseModel):
    user: str = Field(description="The username to fetch the latest tweets from.")
    count: int = Field(default=10, description="Number of tweets to fetch.")


class FetchLatestTweetsTool(BaseTool):
    name: str = "fetch_latest_tweets_tool"
    description: str = "Fetch the latest tweets from a user."
    args_schema: ArgsSchema = FetchLatestTweetsToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.latest_tweets(user=user, count=count)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        user: str,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(user, count, run_manager=run_manager.get_sync())


class FetchTweetsAndRepliesByUserToolInput(BaseModel):
    user: str = Field(description="The username to fetch tweets and replies from.")
    query: Optional[str] = Field(
        default=None, description="Query to filter tweets and replies."
    )
    count: int = Field(default=10, description="Number of tweets and replies to fetch.")


class FetchTweetsAndRepliesByUserTool(BaseTool):
    name: str = "fetch_tweets_and_replies_by_user_tool"
    description: str = "Fetch tweets and replies by a specific user."
    args_schema: ArgsSchema = FetchTweetsAndRepliesByUserToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.tweets_and_replies_by_user(
                user=user, query=query, count=count
            )
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        user: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(user, query, count, run_manager=run_manager.get_sync())


class FetchRepliesByPostToolInput(BaseModel):
    post_id: str = Field(description="The ID of the post to fetch replies for.")
    query: Optional[str] = Field(default=None, description="Query to filter replies.")
    count: int = Field(default=10, description="Number of replies to fetch.")


class FetchRepliesByPostTool(BaseTool):
    name: str = "fetch_replies_by_post_tool"
    description: str = "Fetch replies to a specific post."
    args_schema: ArgsSchema = FetchRepliesByPostToolInput
    return_direct: bool = True

    def _run(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.twitter_replies_post(
                post_id=post_id, query=query, count=count
            )
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(post_id, query, count, run_manager=run_manager.get_sync())


class FetchRetweetsByPostToolInput(BaseModel):
    post_id: str = Field(description="The ID of the post to fetch retweets for.")
    query: Optional[str] = Field(default=None, description="Query to filter retweets.")
    count: int = Field(default=10, description="Number of retweets to fetch.")


class FetchRetweetsByPostTool(BaseTool):
    name: str = "fetch_retweets_by_post_tool"
    description: str = "Fetch retweets of a specific post."
    args_schema: ArgsSchema = FetchRetweetsByPostToolInput
    return_direct: bool = True

    def _run(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.twitter_retweets_post(
                post_id=post_id, query=query, count=count
            )
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(post_id, query, count, run_manager=run_manager.get_sync())


class FetchTwitterUserToolInput(BaseModel):
    user: str = Field(description="The username to fetch information about.")


class FetchTwitterUserTool(BaseTool):
    name: str = "fetch_twitter_user_tool"
    description: str = "Fetch information about a specific Twitter user."
    args_schema: ArgsSchema = FetchTwitterUserToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        api_key = os.getenv("DESEARCH_API_KEY")
        if not api_key:
            raise ValueError("DESEARCH_API_KEY environment variable not set.")

        desearch = Desearch(api_key=api_key)
        try:
            return desearch.tweeter_user(user=user)
        except Exception as e:
            return f"An error occurred while calling Desearch: {str(e)}"

    async def _arun(
        self,
        user: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(user, run_manager=run_manager.get_sync())


# Export all tools
all_tools = [
    DesearchTool,
    BasicWebSearchTool,
    BasicTwitterSearchTool,
    FetchTweetsByUrlsTool,
    FetchTweetsByIdTool,
    FetchLatestTweetsTool,
    FetchTweetsAndRepliesByUserTool,
    FetchRepliesByPostTool,
    FetchRetweetsByPostTool,
    FetchTwitterUserTool,
]
