from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document, HumanMessage

if TYPE_CHECKING:
    from langchain.schema.messages import BaseMessage


def _get_docs(response: Any) -> List[Document]:
    return [
        Document(page_content=doc["snippet"], metadata=doc)
        for doc in response.generation_info["documents"]
    ]


class CohereRagRetriever(BaseRetriever):
    """`ChatGPT plugin` retriever."""

    top_k: int = 3
    """Number of documents to return."""

    connectors: List[Dict] = Field(default_factory=lambda: [{"id": "web-search"}])
    """
    When specified, the model's reply will be enriched with information found by
    querying each of the connectors (RAG). These will be returned as langchain
    documents.

    Currently only accepts {"id": "web-search"}.
    """

    llm: BaseChatModel
    """Cohere ChatModel to use."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        """Allow arbitrary types."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = self.llm.generate(
            messages,
            connectors=self.connectors,
            callbacks=run_manager.get_child(),
            **kwargs,
        ).generations[0][0]
        return _get_docs(res)[: self.top_k]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = (
            await self.llm.agenerate(
                messages,
                connectors=self.connectors,
                callbacks=run_manager.get_child(),
                **kwargs,
            )
        ).generations[0][0]
        return _get_docs(res)[: self.top_k]
