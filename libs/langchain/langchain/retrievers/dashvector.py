"""Wrapper around DashVector vector database."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever


class DashVectorRetriever(BaseRetriever):
    """Retriever that uses DashVector to retrieve documents."""

    embeddings: Embeddings
    collection: Any
    topk: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate python package exists in environment."""
        try:
            import dashvector  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import dashvector python package. "
                "Please install it with `pip install dashvector`."
            )
        return values

    def add_documents(
        self,
        docs: List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        batch_size = 25
        ids = ids or [str(uuid4().hex) for _ in docs]
        embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
        docs_to_upsert = []
        for i, doc in enumerate(docs):
            embedding = embeddings[i]
            metadata = doc.metadata
            metadata["content"] = doc.page_content
            docs_to_upsert.append((ids[i], embedding, metadata))

        for i in range(0, len(docs_to_upsert), batch_size):
            end = min(i + batch_size, len(docs_to_upsert))
            # batch upsert to collection
            ret = self.collection.upsert(docs_to_upsert[i:end])
            if not ret:
                raise Exception(
                    f"Fail to upsert docs to dashvector vector database,"
                    f"Error: {ret.message}"
                )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        filter: Optional[str] = None,
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of relevant documents
        """
        resp = self.collection.query(
            vector=self.embeddings.embed_query(query), topk=self.topk, filter=filter
        )
        if not resp:
            raise Exception(
                f"Fail to query docs from dashvector" f"Error: {resp.message}"
            )

        docs = []
        for doc in resp:
            text = doc.fields.pop("content")
            docs.append(Document(page_content=text, metadata=doc.fields))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
