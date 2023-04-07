"""Wrapper around weaviate vector database."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from pydantic import BaseModel, Extra
from uuid import uuid4

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever


class WeaviateHybridSearchRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    index: Any
    tokenizer: Any
    top_k: int = 4
    alpha: float = 0.5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True    

    # added text_key
    def add_texts(
        self,
        texts: List[str],
        text_key: str
    ) -> None:
        """Upload texts with metadata (properties) to Weaviate."""
        from weaviate.util import get_valid_uuid

        with self._client.batch as batch:
            ids = []
            for i, doc in enumerate(texts):
                data_properties = {
                    self._text_key: doc,
                }
                _id = get_valid_uuid(uuid4())
                batch.add_data_object(data_properties, self._index_name, _id)
                ids.append(_id)
        return ids

    def get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        """Look up similar documents in weaviate."""
        content: Dict[str, Any] = {"concepts": [query]}
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        
        result = query_obj.with_hybrid(content, alpha=self.alpha).with_limit(self.k).do()
        
        docs = []

        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs

    