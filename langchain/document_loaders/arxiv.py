from typing import List, Optional, Iterator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.arxiv import ArxivAPIWrapper


class ArxivLoader(BaseLoader):
    """Loads a query result from arxiv.org into a list of Documents.

    Each document represents one Document.
    The loader converts the original PDF format into the text.
    """

    def __init__(
        self,
        *,
        query: Optional[str] = None,
        load_max_docs: Optional[int] = 100,
        load_all_available_meta: bool = False,
    ):
        """loader with a query and the maximum number of documents to load."""
        self.query = query
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta

    def load(self) -> List[Document]:
        """Loads a query result from arxiv.org into a list of Documents."""
        return list(self.lazy_load(query=query))

    def lazy_load(self) -> Iterator[Document]:
        """Loads a query result from arxiv.org into a list of Documents."""
        arxiv_client = ArxivAPIWrapper(
            load_max_docs=self.load_max_docs,
            load_all_available_meta=self.load_all_available_meta,
        )
        query = query or self.query
        docs = arxiv_client.lazy_load(query=query)
        return docs
