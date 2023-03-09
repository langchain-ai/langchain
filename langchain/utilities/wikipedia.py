"""Util that calls Wikipedia."""
from typing import Any, Dict

from pydantic import BaseModel, Extra, root_validator


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
                "Please it install it with `pip install wikipedia`."
            )
        return values

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        search_results = self.wiki_client.search(query)
        return "\n\n".join(
            filter(
                lambda x: x is not None,
                (
                    self.fetch_formatted_page_summary(search_results[i])
                    for i in range(min(self.top_k_results, len(search_results)))
                ),
            )
        )

    def fetch_formatted_page_summary(self, page: str) -> str:
        try:
            return f"Page: {page}\nSummary: {self.wiki_client.page(title=page).summary}"
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            return None
