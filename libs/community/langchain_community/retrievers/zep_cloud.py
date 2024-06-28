from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

if TYPE_CHECKING:
    from zep_cloud import SessionSearchResult


class ZepCloudRetriever(BaseRetriever):
    """`Zep Cloud` MemoryStore Retriever.

    Search your user's long-term chat history with Zep.

    Zep offers both simple semantic search and Maximal Marginal Relevance (MMR)
    reranking of search results.

    Provide the session_ids or user_id you'd like to retrieve from.
    Note: if you don't provide either,
    Zep will search across all records in your account.

    Args:
        api_key: Your Zep API key
        session_id: The session ID to search. Deprecated. (optional)
        session_ids: The list of sessions which to execute search on (optional)
        user_id: The user whose sessions to search (optional)
        top_k: Number of documents to return (default: 3, optional)
        min_score: Minimum score to return (optional)
        search_type: Type of search to perform (similarity / mmr)
            (default: similarity, optional)
        search_scope: Type of documents to search (messages / facts / summary).
         Defaults to facts
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
    zep_client: Any
    """Zep client used for making API requests."""
    zep_client_async: Any
    """Async Zep client used for making API requests."""
    session_id: Optional[str]
    """Session ID to search. Deprecated."""
    session_ids: Optional[List[str]]
    """List of session IDs to search."""
    user_id: Optional[str]
    """User ID to search."""
    top_k: Optional[int]
    """Number of items to return."""
    min_score: Optional[float]
    """Minimum score to return."""
    search_scope: str = "messages"
    """Which documents to search. Messages, Facts or Summaries?"""
    search_type: str = "similarity"
    """Type of search to perform (similarity / mmr)"""
    mmr_lambda: Optional[float] = None
    """Lambda value for MMR search."""

    @root_validator(pre=True)
    def create_client(cls, values: dict) -> dict:
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
        self, results: List[SessionSearchResult]
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

    def _facts_search_result_to_doc(
        self, results: List[SessionSearchResult]
    ) -> List[Document]:
        return [
            Document(
                page_content=str(r.fact.fact),
                metadata={
                    "score": r.score,
                    "uuid": r.fact.uuid_,
                },
            )
            for r in results or []
            if r.fact
        ]

    def _summary_search_result_to_doc(
        self, results: List[SessionSearchResult]
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

        session_ids = []
        if self.session_id:
            session_ids.append(self.session_id)
        if self.session_ids:
            session_ids.extend(self.session_ids)

        search_output = self.zep_client.memory.search_sessions(
            session_ids=session_ids,
            min_score=self.min_score,
            user_id=self.user_id,
            text=query,
            record_filter=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
            limit=self.top_k,
        )

        if self.search_scope == "summary":
            return self._summary_search_result_to_doc(search_output.results)
        if self.search_scope == "facts":
            return self._facts_search_result_to_doc(search_output.results)

        return self._messages_search_result_to_doc(search_output.results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not self.zep_client_async:
            raise RuntimeError("Zep client not initialized.")

        session_ids = []
        if self.session_id:
            session_ids.append(self.session_id)
        if self.session_ids:
            session_ids.extend(self.session_ids)

        search_output = await self.zep_client_async.memory.search_sessions(
            session_ids=session_ids,
            min_score=self.min_score,
            user_id=self.user_id,
            text=query,
            record_filter=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
            limit=self.top_k,
        )

        if self.search_scope == "summary":
            return self._summary_search_result_to_doc(search_output.results)
        if self.search_scope == "facts":
            return self._facts_search_result_to_doc(search_output.results)

        return self._messages_search_result_to_doc(search_output.results)
