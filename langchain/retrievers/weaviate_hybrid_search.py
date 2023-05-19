"""Wrapper around weaviate vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Extra

from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


class WeaviateHybridSearchRetriever(BaseRetriever):
    def __init__(
        self,
        client: Any,
        index_name: str,
        text_key: str,
        alpha: float = 0.5,
        k: int = 4,
        attributes: Optional[List[str]] = None,
    ):
        try:
            import weaviate
        except ImportError:
            raise ValueError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(client, weaviate.Client):
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )
        self._client = client
        self.k = k
        self.alpha = alpha
        self._index_name = index_name
        self._text_key = text_key
        self._query_attrs = [self._text_key]
        if attributes is not None:
            self._query_attrs.extend(attributes)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    # added text_key
    def add_documents(self, docs: List[Document], **kwargs: Any) -> List[str]:
        """Upload documents to Weaviate."""
        from weaviate.util import get_valid_uuid

        with self._client.batch as batch:
            ids = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata or {}
                data_properties = {self._text_key: doc.page_content, **metadata}

                # If the UUID of one of the objects already exists
                # then the existing objectwill be replaced by the new object.
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                else:
                    _id = get_valid_uuid(uuid4())

                batch.add_data_object(data_properties, self._index_name, _id)
                ids.append(_id)
        return ids

    def get_relevant_documents(
        self, query: str, where_filter: Optional[Dict[str, object]] = None
    ) -> List[Document]:
        """Look up similar documents in Weaviate."""
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if where_filter:
            query_obj = query_obj.with_where(where_filter)

        result = query_obj.with_hybrid(query, alpha=self.alpha).with_limit(self.k).do()
        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")

        docs = []

        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs

    async def aget_relevant_documents(
        self, query: str, where_filter: Optional[Dict[str, object]] = None
    ) -> List[Document]:
        raise NotImplementedError
