"""Util that calls Wikipedia."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator

WIKIPEDIA_MAX_QUERY_LENGTH = 300


class WikipediaAPIWrapper(BaseModel):
    """Wrapper around WikipediaAPI.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results of an input search.
    """

    wiki_client: Any  #: :meta private:
    top_k_results: int = 3

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia

            values["wiki_client"] = wikipedia
        except ImportError:
            raise ValueError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        search_results = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])
        summaries = []
        len_search_results = len(search_results)
        if len_search_results == 0:
            return "No good Wikipedia Search Result was found"
        for i in range(min(self.top_k_results, len_search_results)):
            summary = self.fetch_formatted_page_summary(search_results[i])
            if summary is not None:
                summaries.append(summary)
        return "\n\n".join(summaries)

    def fetch_formatted_page_summary(self, page: str) -> Optional[str]:
        try:
            wiki_page = self.wiki_client.page(title=page, auto_suggest=False)
            return f"Page: {page}\nSummary: {wiki_page.summary}"
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            return None
