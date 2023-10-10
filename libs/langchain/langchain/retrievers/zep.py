from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from zep_python import MemorySearchResult


class ZepRetriever(BaseRetriever):
    """`Zep` long-term memory store retriever.

    Search your user's long-term chat history with Zep.

    Note: You will need to provide the user's `session_id` to use this retriever.

    More on Zep:
    Zep provides long-term conversation storage for LLM apps. The server stores,
    summarizes, embeds, indexes, and enriches conversational AI chat
    histories, and exposes them via simple, low-latency APIs.

    For server installation instructions, see:
    https://docs.getzep.com/deployment/quickstart/
    """

    zep_client: Any
    """Zep client."""
    session_id: str
    """Zep session ID."""
    top_k: Optional[int]
    """Number of documents to return."""

    @root_validator(pre=True)
    def create_client(cls, values: dict) -> dict:
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

    def _search_result_to_doc(
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

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        metadata: Optional[Dict] = None,
    ) -> List[Document]:
        from zep_python import MemorySearchPayload

        payload: MemorySearchPayload = MemorySearchPayload(
            text=query, metadata=metadata
        )

        results: List[MemorySearchResult] = self.zep_client.memory.search_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        metadata: Optional[Dict] = None,
    ) -> List[Document]:
        from zep_python import MemorySearchPayload

        payload: MemorySearchPayload = MemorySearchPayload(
            text=query, metadata=metadata
        )

        results: List[MemorySearchResult] = await self.zep_client.memory.asearch_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)
