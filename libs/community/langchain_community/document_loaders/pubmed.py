from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pubmed import PubMedAPIWrapper


class PubMedLoader(BaseLoader):
    """Load from the `PubMed` biomedical library.

    Attributes:
        query: The query to be passed to the PubMed API.
        load_max_docs: The maximum number of documents to load.
    """

    def __init__(
        self,
        query: str,
        load_max_docs: Optional[int] = 3,
    ):
        """Initialize the PubMedLoader.

        Args:
            query: The query to be passed to the PubMed API.
            load_max_docs: The maximum number of documents to load.
              Defaults to 3.
        """
        self.query = query
        self.load_max_docs = load_max_docs
        self._client = PubMedAPIWrapper(  # type: ignore[call-arg]
            top_k_results=load_max_docs,  # type: ignore[arg-type]
        )

    def lazy_load(self) -> Iterator[Document]:
        for doc in self._client.lazy_load_docs(self.query):
            yield doc
