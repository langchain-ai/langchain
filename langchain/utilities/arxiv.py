"""Util that calls Arxiv."""
from typing import Any, Dict

from pydantic import BaseModel, Extra, root_validator


class ArxivAPIWrapper(BaseModel):
    """Wrapper around ArxivAPI.

    To use, you should have the ``arxiv`` python package installed.
    https://lukasschwab.me/arxiv.py/index.html
    This wrapper will use the Arxiv API to conduct searches and
    fetch document summaries. By default, it will return the document summaries
    of the top-k results of an input search.
    """

    arxiv_client: Any  #: :meta private:
    arxiv_exceptions: Any  # :meta private:
    top_k_results: int = 3
    ARXIV_MAX_QUERY_LENGTH = 300

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import arxiv

            values["arxiv_search"] = arxiv.Search
            values["arxiv_exceptions"] = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            )
        except ImportError:
            raise ValueError(
                "Could not import arxiv python package. "
                "Please install it with `pip install arxiv`."
            )
        return values

    def run(self, query: str) -> str:
        """
        Run Arxiv search and get the document meta information.
        See https://lukasschwab.me/arxiv.py/index.html#Search
        See https://lukasschwab.me/arxiv.py/index.html#Result
        It uses only the most informative fields of document meta information.
        """
        try:
            docs = [
                f"Published: {result.updated.date()}\nTitle: {result.title}\n"
                f"Authors: {', '.join(a.name for a in result.authors)}\n"
                f"Summary: {result.summary}"
                for result in self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
            ]
            return "\n\n".join(docs) if docs else "No good Arxiv Result was found"
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
