"""Retriever wrapper for Raindrop API."""
from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from lm_raindrop import Raindrop, AsyncRaindrop
import os
import json
import uuid

class RaindropRetriever(BaseRetriever, BaseModel):
    """Retriever that uses the Raindrop API for semantic search.
    
    This retriever uses the Raindrop API to perform semantic search across your saved documents.
    For more information about the API, see: https://docs.liquidmetal.ai/

    Example:
        .. code-block:: python

            from raindrop_retriever import RaindropRetriever
            
            # Will try to get API key from RAINDROP_API_KEY env var if not provided
            retriever = RaindropRetriever(api_key="your-api-key")  # or just RaindropRetriever()
            documents = retriever.invoke("your query")  # Use invoke instead of get_relevant_documents

    Args:
        api_key (str, optional): Raindrop API key. If not provided, will try to get from RAINDROP_API_KEY env var.
            You can obtain an API key by signing up at https://raindrop.run
    """

    api_key: str = Field(default_factory=lambda: os.getenv("RAINDROP_API_KEY"))
    client: Raindrop = None
    async_client: AsyncRaindrop = None

    def __init__(self, **kwargs):
        # First call the parent class's __init__ to set up the model
        super().__init__(**kwargs)
        
        # Then check for API key
        api_key = kwargs.get("api_key") or self.api_key
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either through the constructor "
                "or by setting the RAINDROP_API_KEY environment variable."
            )
        
        # Initialize the clients
        self.client = Raindrop(api_key=api_key)
        self.async_client = AsyncRaindrop(api_key=api_key)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: Callbacks to track the run

        Returns:
            List of relevant documents
        """
        # Create a unique request ID for this search
        request_id = str(uuid.uuid4())

        # Perform the chunk search
        search_params = {
            "input": query,
            "request_id": request_id,
        }

        chunk_search = self.client.chunk_search.create(**search_params)

        # Convert results to Documents
        documents = []
        for result in chunk_search.results:
            # Parse source from JSON string if needed
            source = result.source
            if isinstance(source, str):
                try:
                    source = json.loads(source)
                except json.JSONDecodeError:
                    source = {"raw": source}

            # Create Document with text content and metadata
            doc = Document(
                page_content=result.text,
                metadata={
                    "chunk_signature": result.chunk_signature,
                    "payload_signature": result.payload_signature,
                    "score": result.score,
                    "type": result.type,
                    "source": source
                }
            )
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        
        This implementation uses the AsyncRaindrop client for better performance.

        Args:
            query: String to find relevant documents for
            run_manager: Callbacks to track the run

        Returns:
            List of relevant documents
        """
        # Create a unique request ID for this search
        request_id = str(uuid.uuid4())

        # Perform the chunk search using the async client
        search_params = {
            "input": query,
            "request_id": request_id,
        }

        chunk_search = await self.async_client.chunk_search.create(**search_params)

        # Convert results to Documents
        documents = []
        for result in chunk_search.results:
            # Parse source from JSON string if needed
            source = result.source
            if isinstance(source, str):
                try:
                    source = json.loads(source)
                except json.JSONDecodeError:
                    source = {"raw": source}

            # Create Document with text content and metadata
            doc = Document(
                page_content=result.text,
                metadata={
                    "chunk_signature": result.chunk_signature,
                    "payload_signature": result.payload_signature,
                    "score": result.score,
                    "type": result.type,
                    "source": source
                }
            )
            documents.append(doc)

        return documents 