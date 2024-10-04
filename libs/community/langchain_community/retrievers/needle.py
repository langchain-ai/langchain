from typing import List, Optional
from needle.v1 import NeedleClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, ConfigDict

class NeedleRetriever(BaseRetriever, BaseModel):
    """NeedleRetriever.

    Retrieves relevant documents or context from a Needle collection based on a search query.
    """

    needle_api_key: Optional[str] = Field(None, description="Needle API Key")
    collection_id: Optional[str] = Field(..., description="The ID of the Needle collection to search in")
    client: Optional[NeedleClient] = None

    def _initialize_client(self) -> None:
        """Initialize NeedleClient with the provided API key."""
        if not self.client:
            self.client = NeedleClient(api_key=self.needle_api_key)

    def _search_collection(self, query: str) -> List[Document]:
        """Search the Needle collection for relevant documents."""
        self._initialize_client()
        results = self.client.collections.search(collection_id=self.collection_id, text=query)
        docs = [Document(page_content=result.content) for result in results]
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents based on the query."""
        return self._search_collection(query)
