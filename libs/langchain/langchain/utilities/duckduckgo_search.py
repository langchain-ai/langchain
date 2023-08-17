"""Util that calls DuckDuckGo Search.

No setup required. Free.
https://pypi.org/project/duckduckgo-search/
"""
from typing import Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator


class DuckDuckGoSearchAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup.
    """

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
            from duckduckgo_search import DDGS  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import duckduckgo-search python package. "
                "Please install it with `pip install duckduckgo-search`."
            )
        return values

    def get_snippets(self, query: str) -> List[str]:
        """Run query through DuckDuckGo and return concatenated results."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
            )
            if results is None:
                return ["No good DuckDuckGo Search Result was found"]
            snippets = []
            for i, res in enumerate(results, 1):
                if res is not None:
                    snippets.append(res["body"])
                if len(snippets) == self.max_results:
                    break
        return snippets

    def run(self, query: str) -> str:
        snippets = self.get_snippets(query)
        return " ".join(snippets)

    def results(
        self, query: str, num_results: int, backend: str = "api"
    ) -> List[Dict[str, str]]:
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
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                backend=backend,
            )
            if results is None:
                return [{"Result": "No good DuckDuckGo Search Result was found"}]

            def to_metadata(result: Dict) -> Dict[str, str]:
                if backend == "news":
                    return {
                        "date": result["date"],
                        "title": result["title"],
                        "snippet": result["body"],
                        "source": result["source"],
                        "link": result["url"],
                    }
                return {
                    "snippet": result["body"],
                    "title": result["title"],
                    "link": result["href"],
                }

            formatted_results = []
            for i, res in enumerate(results, 1):
                if res is not None:
                    formatted_results.append(to_metadata(res))
                if len(formatted_results) == num_results:
                    break
        return formatted_results
