from typing import List, Optional

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
        query: str,
        load_max_docs: Optional[int] = 100,
        load_all_available_meta: Optional[bool] = False,
    ):
        self.query = query
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta

    def load(self) -> List[Document]:
        arxiv_client = ArxivAPIWrapper(
            load_max_docs=self.load_max_docs,
            load_all_available_meta=self.load_all_available_meta,
        )
        docs = arxiv_client.load(self.query)
        return docs
