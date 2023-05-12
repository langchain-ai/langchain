from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from zep_python import SearchResult


class ZepRetriever(BaseRetriever):
    """A Retriever implementation for the Zep long-term memory store. Search your
    user's long-term chat history with Zep.

    Note: You will need to provide the user's `session_id` to use this retriever.

    More on Zep:
    Zep provides long-term conversation storage for LLM apps. The server stores,
    summarizes, embeds, indexes, and enriches conversational AI chat
    histories, and exposes them via simple, low-latency APIs.

    For server installation instructions, see: https://github.com/getzep/zep
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

    def _search_result_to_doc(self, results: List[SearchResult]) -> List[Document]:
        return [
            Document(
                page_content=r.message["content"],
                metadata={
                    "source": r.message["uuid"],
                    "score": r.dist,
                    "role": r.message["role"],
                    "token_count": r.message["token_count"],
                    "created_at": r.message["created_at"],
                },
            )
            for r in results
            if r.message
        ]

    def get_relevant_documents(self, query: str) -> List[Document]:
        from zep_python import SearchPayload, SearchResult

        payload: SearchPayload = SearchPayload(text=query)

        results: List[SearchResult] = self.zep_client.search_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        from zep_python import SearchPayload, SearchResult

        payload: SearchPayload = SearchPayload(text=query)

        results: List[SearchResult] = await self.zep_client.asearch_memory(
            self.session_id, payload, limit=self.top_k
        )

        return self._search_result_to_doc(results)
