from typing import List, Optional
from pydantic import Field
import requests
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class RAGProRetriever(BaseRetriever):
    """RAG Pro API retriever."""

    api_key: str = Field(..., description="The API key for authentication.")
    base_url: str = Field(
        "https://2q5vp3y0d0.execute-api.us-east-1.amazonaws.com/dev",
        description="The base URL for the API."
    )
    model: str = Field("small", description="The model to use for retrieval.")
    include_sources: bool = Field(True, description="Whether to include sources in the response.")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, urls: Optional[List[str]] = None
    ) -> List[Document]:
        """Get documents relevant to a query."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

        params = {
            "query": query,
            "model": self.model,
            "sources": str(self.include_sources).lower()
        }

        data = {
            "queryStringParameters": params,
            "multiValueQueryStringParameters": {"urls": urls} if urls else {}
        }

        response = requests.post(f"{self.base_url}/retrieve", headers=headers, json=data, timeout=40)
        response.raise_for_status()

        results = response.json()
        documents = []

        for result in results:
            doc = Document(
                page_content=result["page_content"],
                metadata=result["metadata"]
            )
            documents.append(doc)

        return documents
