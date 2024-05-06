from typing import Any, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.arxiv import ArxivAPIWrapper


class ArxivLoader(BaseLoader):
    """Load a query result from `Arxiv`.

    The loader converts the original PDF format into the text.

    Args:
        Supports all arguments of `ArxivAPIWrapper`.
    """

    def __init__(
        self, query: str, doc_content_chars_max: Optional[int] = None, **kwargs: Any
    ):
        self.query = query
        self.client = ArxivAPIWrapper(
            doc_content_chars_max=doc_content_chars_max, **kwargs
        )

    def lazy_load(self) -> Iterator[Document]:
        yield from self.client.lazy_load(self.query)

    def get_summaries_as_docs(self) -> List[Document]:
        return self.client.get_summaries_as_docs(self.query)
