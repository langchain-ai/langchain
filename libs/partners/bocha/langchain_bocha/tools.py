import os
import requests
from typing import Any, Optional
from langchain_core.tools import BaseTool

BOCHA_SEARCH_URL = "https://api.bocha.ai/v1/web-search"


class BochaSearchRun(BaseTool):
    """Tool that searches the web using Bocha AI.

    Example:
        .. code-block:: python

            from langchain_bocha import BochaSearchRun

            search = BochaSearchRun()
            results = search.run("Beijing weather")
    """

    name: str = "bocha_search"
    description: str = "Search the web using Bocha AI. Input should be a search query string."
    api_key: Optional[str] = None

    def _run(self, query: str, **kwargs: Any) -> str:
        key = self.api_key or os.environ.get("BOCHA_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"}
        params = {"q": query, "count": 5}
        resp = requests.get(BOCHA_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        results = resp.json().get("webPages", {}).get("value", [])
        return "\n\n".join(
            f"{r['name']}: {r['snippet']}" for r in results
        )


class BochaSearchResults(BaseTool):
    """Tool that returns raw JSON search results from Bocha AI.

    Example:
        .. code-block:: python

            from langchain_bocha import BochaSearchResults

            search = BochaSearchResults()
            results = search.run("Beijing weather")
    """

    name: str = "bocha_search_results"
    description: str = "Search the web using Bocha AI. Returns raw JSON results."
    api_key: Optional[str] = None

    def _run(self, query: str, **kwargs: Any) -> dict:
        key = self.api_key or os.environ.get("BOCHA_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"}
        params = {"q": query, "count": 5}
        resp = requests.get(BOCHA_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()
