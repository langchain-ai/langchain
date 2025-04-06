from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.perplexity import PerplexityWrapper


class Perplexity(BaseTool):
    """Tool that queries the Perplexity API for chat completions.

    This tool leverages Perplexity's generative search capabilities powered by
    Sonar and Sonar Pro models. It returns real-time, web-connected research with citations.
    Input should be a user query.
    """

    name: str = "perplexity"
    description: str = (
        "A tool for querying the Perplexity API's generative search capabilities. "
        "Useful for answering queries with real-time research and citations. "
        "Input should be a user query."
    )
    perplexity_wrapper: PerplexityWrapper

    @classmethod
    def from_api_key(
        cls, api_key: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> Perplexity:
        """Create a tool from an API key.

        Args:
            api_key: The API key to use.
            search_kwargs: Additional kwargs to pass to the Perplexity API.
            **kwargs: Additional kwargs to pass to the tool.

        Returns:
            An instance of the Perplexity tool.
        """
        wrapper = PerplexityWrapper(api_key=api_key, search_kwargs=search_kwargs or {})
        return cls(perplexity_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool to query the Perplexity API."""
        return self.perplexity_wrapper.run(query)
