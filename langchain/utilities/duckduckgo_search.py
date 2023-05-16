"""Util that calls DuckDuckGo Search.

No setup required. Free.
https://pypi.org/project/duckduckgo-search/
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra
from pydantic.class_validators import root_validator


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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            from duckduckgo_search import ddg  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import duckduckgo-search python package. "
                "Please install it with `pip install duckduckgo-search`."
            )
        return values

    def get_snippets(self, query: str) -> List[str]:
        """Run query through DuckDuckGo and return concatenated results."""
        from duckduckgo_search import ddg

        results = ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            time=self.time,
            max_results=self.max_results,
        )
        if results is None or len(results) == 0:
            return ["No good DuckDuckGo Search Result was found"]
        snippets = [result["body"] for result in results]
        return snippets

    def run(self, query: str) -> str:
        snippets = self.get_snippets(query)
        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict[str, str]]:
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

        results = ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            time=self.time,
            max_results=num_results,
        )

        if results is None or len(results) == 0:
            return [{"Result": "No good DuckDuckGo Search Result was found"}]

        def to_metadata(result: Dict) -> Dict[str, str]:
            return {
                "snippet": result["body"],
                "title": result["title"],
                "link": result["href"],
            }

        return [to_metadata(result) for result in results]
