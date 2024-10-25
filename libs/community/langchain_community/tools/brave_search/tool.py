from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.brave_search import BraveSearchWrapper


class BraveSearch(BaseTool):  # type: ignore[override]
    """Tool that queries the BraveSearch."""

    name: str = "brave_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    search_wrapper: BraveSearchWrapper

    @classmethod
    def from_api_key(
        cls, api_key: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> BraveSearch:
        """Create a tool from an api key.

        Args:
            api_key: The api key to use.
            search_kwargs: Any additional kwargs to pass to the search wrapper.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        wrapper = BraveSearchWrapper(api_key=api_key, search_kwargs=search_kwargs or {})
        return cls(search_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.search_wrapper.run(query)
