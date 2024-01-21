"""Tool for the Stack Exchange API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper


class StackExchangeToolInput(BaseModel):
    """Input for the SemanticScholar tool."""

    query: str = Field(description="Stack Question to look up")


class StackExchangeTool(BaseTool):
    """Tool that uses StackExchange"""

    name: str = "StackExchange"
    description: str = (
        "A wrapper around StackExchange. "
        "Useful for when you need to answer specific programming questions"
        "code excerpts, code examples and solutions"
        "Input should be a fully formed question."
    )
    api_wrapper: StackExchangeAPIWrapper
    args_schema: Type[BaseModel] = StackExchangeToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Stack Exchange tool."""
        return self.api_wrapper.run(query)
