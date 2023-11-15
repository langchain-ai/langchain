from typing import Iterator, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.brave_search import BraveSearchWrapper


class BraveSearchLoader(BaseLoader):
    """Load with `Brave Search` engine."""

    def __init__(self, query: str, api_key: str, search_kwargs: Optional[dict] = None):
        """Initializes the BraveLoader.

        Args:
            query: The query to search for.
            api_key: The API key to use.
            search_kwargs: The search kwargs to use.
        """
        self.query = query
        self.api_key = api_key
        self.search_kwargs = search_kwargs or {}

    def load(self) -> List[Document]:
        brave_client = BraveSearchWrapper(
            api_key=self.api_key,
            search_kwargs=self.search_kwargs,
        )
        return brave_client.download_documents(self.query)

    def lazy_load(self) -> Iterator[Document]:
        for doc in self.load():
            yield doc
