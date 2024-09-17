from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import model_validator

if TYPE_CHECKING:
    from zep_cloud import MemorySearchResult, SearchScope, SearchType
    from zep_cloud.client import AsyncZep, Zep


class ZepCloudRetriever(BaseRetriever):
    """`Zep Cloud` MemoryStore Retriever.

    Search your user's long-term chat history with Zep.

    Zep offers both simple semantic search and Maximal Marginal Relevance (MMR)
    reranking of search results.

    Note: You will need to provide the user's `session_id` to use this retriever.

    Args:
        api_key: Your Zep API key
        session_id: Identifies your user or a user's session (required)
        top_k: Number of documents to return (default: 3, optional)
        search_type: Type of search to perform (similarity / mmr)
            (default: similarity, optional)
        mmr_lambda: Lambda value for MMR search. Defaults to 0.5 (optional)

    Zep - Recall, understand, and extract data from chat histories.
    Power personalized AI experiences.
    =========
    Zep is a long-term memory service for AI Assistant apps.
    With Zep, you can provide AI assistants with the ability
    to recall past conversations,
    no matter how distant, while also reducing hallucinations, latency, and cost.

    see Zep Cloud Docs: https://help.getzep.com
    """

    api_key: str
    """Your Zep API key."""
    zep_client: Zep
    """Zep client used for making API requests."""
    zep_client_async: AsyncZep
    """Async Zep client used for making API requests."""
    session_id: str
    """Zep session ID."""
    top_k: Optional[int]
    """Number of items to return."""
    search_scope: SearchScope = "messages"
    """Which documents to search. Messages or Summaries?"""
    search_type: SearchType = "similarity"
    """Type of search to perform (similarity / mmr)"""
    mmr_lambda: Optional[float] = None
    """Lambda value for MMR search."""

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: dict) -> Any:
        try:
            from zep_cloud.client import AsyncZep, Zep
        except ImportError:
            raise ImportError(
                "Could not import zep-cloud package. "
                "Please install it with `pip install zep-cloud`."
            )
        if values.get("api_key") is None:
            raise ValueError("Zep API key is required.")
        values["zep_client"] = Zep(api_key=values.get("api_key"))
        values["zep_client_async"] = AsyncZep(api_key=values.get("api_key"))
        return values

    def _messages_search_result_to_doc(
        self, results: List[MemorySearchResult]
    ) -> List[Document]:
        return [
            Document(
                page_content=str(r.message.content),
                metadata={
                    "score": r.score,
                    "uuid": r.message.uuid_,
                    "created_at": r.message.created_at,
                    "token_count": r.message.token_count,
                    "role": r.message.role or r.message.role_type,
                },
            )
            for r in results or []
            if r.message
        ]

    def _summary_search_result_to_doc(
        self, results: List[MemorySearchResult]
    ) -> List[Document]:
        return [
            Document(
                page_content=str(r.summary.content),
                metadata={
                    "score": r.score,
                    "uuid": r.summary.uuid_,
                    "created_at": r.summary.created_at,
                    "token_count": r.summary.token_count,
                },
            )
            for r in results
            if r.summary
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not self.zep_client:
            raise RuntimeError("Zep client not initialized.")

        results = self.zep_client.memory.search(
            self.session_id,
            text=query,
            metadata=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
            limit=self.top_k,
        )

        if self.search_scope == "summary":
            return self._summary_search_result_to_doc(results)

        return self._messages_search_result_to_doc(results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not self.zep_client_async:
            raise RuntimeError("Zep client not initialized.")

        results = await self.zep_client_async.memory.search(
            self.session_id,
            text=query,
            metadata=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
            limit=self.top_k,
        )

        if self.search_scope == "summary":
            return self._summary_search_result_to_doc(results)

        return self._messages_search_result_to_doc(results)
