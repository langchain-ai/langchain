"""Util that calls DuckDuckGo Search.

No setup required. Free.
https://pypi.org/project/duckduckgo-search/
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra


class DuckDuckGoSearchAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup
    """

    k: int = 10
    region: Optional[str] = "wt-wt"
    safesearch: str = "moderate"
    time: Optional[str] = "y"
    max_results: int = 5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def run(self, query: str) -> str:
        from duckduckgo_search import ddg

        """Run query through DuckDuckGo and return results."""
        results = ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            time=self.time,
            max_results=self.max_results,
        )
        snippets = []

        if len(results) == 0:
            return "No good DuckDuckGo Search Result was found"
        for result in results:
            snippets.append(result["body"])
        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        from duckduckgo_search import ddg

        metadata_results = []
        results = ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            time=self.time,
            max_results=num_results,
        )

        if len(results) == 0:
            return [{"Result": "No good DuckDuckGo Search Result was found"}]
        for result in results:
            metadata_result = {
                "snippet": result["body"],
                "title": result["title"],
                "link": result["href"],
            }
            metadata_results.append(metadata_result)

        return metadata_results
