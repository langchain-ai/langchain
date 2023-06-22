from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from zep_python import MemorySearchResult


class ZepRetriever(BaseRetriever):
    """A Retriever implementation for the Zep long-term memory store. Search your
    user's long-term chat history with Zep.

    Note: You will need to provide the user's `session_id` to use this retriever.

    More on Zep:
    Zep provides long-term conversation storage for LLM apps. The server stores,
    summarizes, embeds, indexes, and enriches conversational AI chat
    histories, and exposes them via simple, low-latency APIs.

    For server installation instructions, see:
    https://getzep.github.io/deployment/quickstart/
    """

    def __init__(
        self,
        session_id: str,
        url: str,
        top_k: Optional[int] = None,
    ):
        try:
            from zep_python import ZepClient
        except ImportError:
            raise ValueError(
                "Could not import zep-python package. "
                "Please install it with `pip install zep-python`."
            )

        self.zep_client = ZepClient(base_url=url)
        self.session_id = session_id
        self.top_k = top_k

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

    def get_relevant_documents(
        self, query: str, metadata: Optional[Dict] = None
    ) -> List[Document]:
        from zep_python import MemorySearchPayload

        payload: MemorySearchPayload = MemorySearchPayload(
            text=query, metadata=metadata
        )

        results: List[MemorySearchResult] = self.zep_client.search_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)

    async def aget_relevant_documents(
        self, query: str, metadata: Optional[Dict] = None
    ) -> List[Document]:
        from zep_python import MemorySearchPayload

        payload: MemorySearchPayload = MemorySearchPayload(
            text=query, metadata=metadata
        )

        results: List[MemorySearchResult] = await self.zep_client.asearch_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)
