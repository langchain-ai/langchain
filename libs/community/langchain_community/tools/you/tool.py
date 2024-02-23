from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.retrievers.you import YouSearchAPIWrapper


class YouInput(BaseModel):
    query: str = Field(description="should be a search query")


class YouSearchTool(BaseTool):
    """Tool that searches the you.com API"""

    name = "you_search"
    description = (
        "The YOU APIs make LLMs and search experiences more factual and"
        "up to date with realtime web data."
    )
    args_schema: Type[BaseModel] = YouInput
    api_wrapper: YouSearchAPIWrapper = Field(default_factory=YouSearchAPIWrapper)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the you.com tool."""
        return self.api_wrapper.results(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the you.com tool asynchronously."""
        return await self.api_wrapper.results_async(query)
