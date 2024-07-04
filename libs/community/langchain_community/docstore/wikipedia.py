"""Wrapper around wikipedia API."""

from typing import Union

from langchain_core.documents import Document

from langchain_community.docstore.base import Docstore


class Wikipedia(Docstore):
    """Wikipedia API."""

    def __init__(self) -> None:
        """Check that wikipedia package is installed."""
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )

    def search(self, search: str) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            search: search string.

        Returns: a Document object or error message.
        """
        import wikipedia

        try:
            page_content = wikipedia.page(search).content
            url = wikipedia.page(search).url
            result: Union[str, Document] = Document(
                page_content=page_content, metadata={"page": url}
            )
        except wikipedia.PageError:
            result = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
        except wikipedia.DisambiguationError:
            result = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
        return result
