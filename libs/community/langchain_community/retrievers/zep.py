from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import model_validator

if TYPE_CHECKING:
    from zep_python.memory import MemorySearchResult


class SearchScope(str, Enum):
    """Which documents to search. Messages or Summaries?"""

    messages = "messages"
    """Search chat history messages."""
    summary = "summary"
    """Search chat history summaries."""


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class ZepRetriever(BaseRetriever):
    """`Zep` MemoryStore Retriever.

    Search your user's long-term chat history with Zep.

    Zep offers both simple semantic search and Maximal Marginal Relevance (MMR)
    reranking of search results.

    Note: You will need to provide the user's `session_id` to use this retriever.

    Args:
        url: URL of your Zep server (required)
        api_key: Your Zep API key (optional)
        session_id: Identifies your user or a user's session (required)
        top_k: Number of documents to return (default: 3, optional)
        search_type: Type of search to perform (similarity / mmr) (default: similarity,
                                                                    optional)
        mmr_lambda: Lambda value for MMR search. Defaults to 0.5 (optional)

    Zep - Fast, scalable building blocks for LLM Apps
    =========
    Zep is an open source platform for productionizing LLM apps. Go from a prototype
    built in LangChain or LlamaIndex, or a custom app, to production in minutes without
    rewriting code.

    For server installation instructions, see:
    https://docs.getzep.com/deployment/quickstart/
    """

    zep_client: Optional[Any] = None
    """Zep client."""
    url: str
    """URL of your Zep server."""
    api_key: Optional[str] = None
    """Your Zep API key."""
    session_id: str
    """Zep session ID."""
    top_k: Optional[int]
    """Number of items to return."""
    search_scope: SearchScope = SearchScope.messages
    """Which documents to search. Messages or Summaries?"""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""
    mmr_lambda: Optional[float] = None
    """Lambda value for MMR search."""

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: dict) -> Any:
        try:
            from zep_python import ZepClient
        except ImportError:
            raise ImportError(
                "Could not import zep-python package. "
                "Please install it with `pip install zep-python`."
            )
        values["zep_client"] = values.get(
            "zep_client",
            ZepClient(base_url=values["url"], api_key=values.get("api_key")),
        )
        return values

    def _messages_search_result_to_doc(
        self, results: List[MemorySearchResult]
    ) -> List[Document]:
        return [
            Document(
                page_content=r.message.pop("content"),
                metadata={"score": r.dist, **r.message},
            )
            for r in results
            if r.message
        ]

    def _summary_search_result_to_doc(
        self, results: List[MemorySearchResult]
    ) -> List[Document]:
        return [
            Document(
                page_content=r.summary.content,
                metadata={
                    "score": r.dist,
                    "uuid": r.summary.uuid,
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
        from zep_python.memory import MemorySearchPayload

        if not self.zep_client:
            raise RuntimeError("Zep client not initialized.")

        payload = MemorySearchPayload(
            text=query,
            metadata=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
        )

        results: List[MemorySearchResult] = self.zep_client.memory.search_memory(
            self.session_id, payload, limit=self.top_k
        )

        if self.search_scope == SearchScope.summary:
            return self._summary_search_result_to_doc(results)

        return self._messages_search_result_to_doc(results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        from zep_python.memory import MemorySearchPayload

        if not self.zep_client:
            raise RuntimeError("Zep client not initialized.")

        payload = MemorySearchPayload(
            text=query,
            metadata=metadata,
            search_scope=self.search_scope,
            search_type=self.search_type,
            mmr_lambda=self.mmr_lambda,
        )

        results: List[MemorySearchResult] = await self.zep_client.memory.asearch_memory(
            self.session_id, payload, limit=self.top_k
        )

        if self.search_scope == SearchScope.summary:
            return self._summary_search_result_to_doc(results)

        return self._messages_search_result_to_doc(results)
