import os
from typing import Any, List, Literal, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DappierRetriever(BaseRetriever):
    """Dappier retriever."""

    data_model_id: str
    """Data model ID, starting with dm_."""
    k: int = 9
    """Number of documents to return."""
    ref: Optional[str] = None
    """Site domain where AI recommendations are displayed."""
    num_articles_ref: int = 0
    """Minimum number of articles from the ref domain specified.
    The rest will come from other sites within the RAG model."""
    search_algorithm: Literal[
        "most_recent", "most_recent_semantic", "semantic", "trending"
    ] = "most_recent"
    """Search algorithm for retrieving articles."""
    api_key: Optional[str] = None
    """The API key used to interact with the Dappier APIs."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callabacks handler to use
        Returns:
            List of relevant documents
        """
        try:
            from dappier import Dappier
        except ImportError:
            raise ImportError(
                "Dappier python package not found."
                "Please install it with `pip install dappier`"
            )
        try:
            if not self.data_model_id:
                raise ValueError("Data model id is not initialized.")
            dp_client = Dappier(api_key=self.api_key or os.environ["DAPPIER_API_KEY"])
            response = dp_client.get_ai_recommendations(
                query=query,
                data_model_id=self.data_model_id,
                similarity_top_k=self.k,
                ref=self.ref,
                num_articles_ref=self.num_articles_ref,
                search_algorithm=self.search_algorithm,
            )
            return self._extract_documents(response=response)
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant docuements for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        try:
            from dappier import DappierAsync
        except ImportError:
            raise ImportError(
                "Dappier python package not found."
                "Please install it with `pip install dappier`."
            )
        try:
            dp_client = DappierAsync(
                api_key=self.api_key or os.environ["DAPPIER_API_KEY"]
            )
            async with dp_client as client:
                response = await client.get_ai_recommendations_async(
                    query=query,
                    data_model_id=self.data_model_id,
                    similarity_top_k=self.k,
                    ref=self.ref,
                    num_articles_ref=self.num_articles_ref,
                    search_algorithm=self.search_algorithm,
                )
                return self._extract_documents(response=response)
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e

    def _extract_documents(self, response: Any) -> List[Document]:
        """Extract documents from an api response"""

        from dappier.types import AIRecommendationsResponse

        docs: List[Document] = []
        rec_response: AIRecommendationsResponse = response
        if rec_response.response is None or rec_response.response.results is None:
            return docs
        for doc in rec_response.response.results:
            docs.append(
                Document(
                    page_content=doc.summary,
                    metadata={
                        "title": doc.title,
                        "author": doc.author,
                        "source_url": doc.source_url,
                        "image_url": doc.image_url,
                        "pubdata": doc.pubdate,
                    },
                )
            )
        return docs
