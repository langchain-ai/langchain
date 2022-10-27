"""Wrapper around wikipedia API."""


from typing import Optional, Tuple

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document


class Wikipedia(Docstore):
    """Wrapper around wikipedia API."""

    def __init__(self) -> None:
        """Check that wikipedia package is installed."""
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import wikipedia python package. "
                "Please it install it with `pip install wikipedia`."
            )

    def search(self, search: str) -> Tuple[str, Optional[Document]]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.
        """
        import wikipedia

        try:
            page_content = wikipedia.page(search).content
            wiki_page = Document(page_content=page_content)
            observation = wiki_page.summary
        except wikipedia.PageError:
            wiki_page = None
            observation = (
                f"Could not find [{search}]. " f"Similar: {wikipedia.search(search)}"
            )
        except wikipedia.DisambiguationError:
            wiki_page = None
            observation = (
                f"Could not find [{search}]. " f"Similar: {wikipedia.search(search)}"
            )
        return observation, wiki_page
