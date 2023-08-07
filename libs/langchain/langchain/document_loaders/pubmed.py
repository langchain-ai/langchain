from typing import Iterator, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.pubmed import PubMedAPIWrapper


class PubMedLoader(BaseLoader):
    """Loads a query result from PubMed biomedical library into a list of Documents."""

    def __init__(
        self,
        query: str,
        load_max_docs: Optional[int] = 3,
    ):
        self.query = query
        """The query to be passed to the PubMed API."""
        self.load_max_docs = load_max_docs
        """The maximum number of documents to load."""
        self._client = PubMedAPIWrapper(
            top_k_results=load_max_docs,
        )

    def load(self) -> List[Document]:
        return self._client.load_docs(self.query)

    def lazy_load(self) -> Iterator[Document]:
        for doc in self._client.lazy_load_docs(self.query):
            yield doc
