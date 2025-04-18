from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr

from langchain_community.utilities.brave_search import BraveSearchWrapper


class BraveSearch(BaseTool):
    """Tool that queries the BraveSearch.

    Api key can be provided as an environment variable BRAVE_SEARCH_API_KEY
    or as a parameter.


    Example usages:
    .. code-block:: python
        # uses BRAVE_SEARCH_API_KEY from environment
        tool = BraveSearch()

    .. code-block:: python
        # uses the provided api key
        tool = BraveSearch.from_api_key("your-api-key")

    .. code-block:: python
        # uses the provided api key and search kwargs
        tool = BraveSearch.from_api_key(
                                api_key = "your-api-key",
                                search_kwargs={"max_results": 5}
                                )

    .. code-block:: python
        # uses BRAVE_SEARCH_API_KEY from environment
        tool = BraveSearch.from_search_kwargs({"max_results": 5})
    """

    name: str = "brave_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    search_wrapper: BraveSearchWrapper = Field(default_factory=BraveSearchWrapper)

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
        wrapper = BraveSearchWrapper(
            api_key=SecretStr(api_key), search_kwargs=search_kwargs or {}
        )
        return cls(search_wrapper=wrapper, **kwargs)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> BraveSearch:
        """Create a tool from search kwargs.

        Uses the environment variable BRAVE_SEARCH_API_KEY for api key.

        Args:
            search_kwargs: Any additional kwargs to pass to the search wrapper.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        # we can not provide api key because it's calculated in the wrapper,
        # so the ignore is needed for linter
        # not ideal but needed to keep the tool code changes non-breaking
        wrapper = BraveSearchWrapper(search_kwargs=search_kwargs)
        return cls(search_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.search_wrapper.run(query)
