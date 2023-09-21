from typing import List, Optional, Any

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.arxiv import ArxivAPIWrapper


class ArxivLoader(BaseLoader):
    """Load a query result from `Arxiv`.

    The loader converts the original PDF format into the text.
    """

    def __init__(
        self,
        query: str,
        **kwargs: Any
    ):
        self.query=query
        self.client = ArxivAPIWrapper(**kwargs)


    def load(self) -> List[Document]:
        return self.client.load(self.query)
