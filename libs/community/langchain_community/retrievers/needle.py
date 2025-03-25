from typing import Any, List, Optional  # noqa: I001

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field


class NeedleRetriever(BaseRetriever, BaseModel):
    """
    NeedleRetriever retrieves relevant documents or context from a Needle collection
    based on a search query.

    Setup:
        Install the `needle-python` library and set your Needle API key.

        .. code-block:: bash

            pip install needle-python
            export NEEDLE_API_KEY="your-api-key"

    Key init args:
        - `needle_api_key` (Optional[str]): The API key for authenticating with Needle.
        - `collection_id` (str): The ID of the Needle collection to search in.
        - `client` (Optional[NeedleClient]): An optional instance of the NeedleClient.
        - `top_k` (Optional[int]): Maximum number of results to return.

    Usage:
        .. code-block:: python

            from langchain_community.retrievers.needle import NeedleRetriever

            retriever = NeedleRetriever(
                needle_api_key="your-api-key",
                collection_id="your-collection-id",
                top_k=10  # optional
            )

            results = retriever.retrieve("example query")
            for doc in results:
                print(doc.page_content)
    """

    client: Optional[Any] = None
    """Optional instance of NeedleClient."""
    needle_api_key: Optional[str] = Field(None, description="Needle API Key")
    collection_id: Optional[str] = Field(
        ..., description="The ID of the Needle collection to search in"
    )
    top_k: Optional[int] = Field(
        default=None, description="Maximum number of search results to return"
    )

    def _initialize_client(self) -> None:
        """
        Initialize the NeedleClient with the provided API key.

        If a client instance is already provided, this method does nothing.
        """
        try:
            from needle.v1 import NeedleClient
        except ImportError:
            raise ImportError("Please install with `pip install needle-python`.")

        if not self.client:
            self.client = NeedleClient(api_key=self.needle_api_key)

    def _search_collection(self, query: str) -> List[Document]:
        """
        Search the Needle collection for relevant documents.

        Args:
            query (str): The search query used to find relevant documents.

        Returns:
            List[Document]: A list of documents matching the search query.
        """
        self._initialize_client()
        if self.client is None:
            raise ValueError("NeedleClient is not initialized. Provide an API key.")

        results = self.client.collections.search(
            collection_id=self.collection_id, text=query, top_k=self.top_k
        )
        docs = [Document(page_content=result.content) for result in results]
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): The query string used to search the collection.
        Returns:
            List[Document]: A list of documents relevant to the query.
        """
        # The `run_manager` parameter is included to match the superclass signature,
        # but it is not used in this implementation.
        return self._search_collection(query)
