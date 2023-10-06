from __future__ import annotations

from typing import Any, Dict, List

from cohere.client import Chat

from langchain.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)
from langchain.chat_models import ChatCohere
from langchain.pydantic_v1 import Field
from langchain.schema import (
    BaseRetriever,
    Document,
    HumanMessage,
)


def _get_docs(response: Chat) -> List[Document]:
    return [
        Document(page_content=doc["snippet"], metadata=doc)
        for doc in response.documents
    ]


class CohereRagRetriever(BaseRetriever, ChatCohere):
    """`ChatGPT plugin` retriever."""

    top_k: int = 3
    """Number of documents to return."""

    connectors: List[Dict] = Field(default=[{"id": "web-search"}])
    """Connectors to use for the search."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        """Allow arbitrary types."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        request = self.get_cohere_chat_request(
            [HumanMessage(content=query)],
            connectors=self.connectors,
            **kwargs,
        )
        response = self.client.chat(**request)
        return _get_docs(response)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        request = self.get_cohere_chat_request(
            [HumanMessage(content=query)],
            connectors=self.connectors,
            **kwargs,
        )
        response = await self.async_client.chat(**request)
        return _get_docs(response)
