from __future__ import annotations

from typing import Any, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.brave_search import BraveSearchWrapper


class BraveSearch(BaseTool):
    name = "brave-search"
    description = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    search_wrapper: BraveSearchWrapper

    @classmethod
    def from_api_key(
        cls, api_key: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> BraveSearch:
        wrapper = BraveSearchWrapper(api_key=api_key, search_kwargs=search_kwargs or {})
        return cls(search_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.search_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BraveSearch does not support async")
