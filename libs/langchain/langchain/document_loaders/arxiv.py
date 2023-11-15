from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.arxiv import ArxivAPIWrapper


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

    def load(self) -> List[Document]:
        return self.client.load(self.query)
