"""Tool for the Wikipedia API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaQueryInput(BaseModel):
    """Input for the WikipediaQuery tool."""

    query: str = Field(description="query to look up on wikipedia")


class WikipediaQueryRun(BaseTool):
    """Tool that searches the Wikipedia API."""

    name: str = "wikipedia"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )
    api_wrapper: WikipediaAPIWrapper

    args_schema: Type[BaseModel] = WikipediaQueryInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool."""
        return self.api_wrapper.run(query)
