"""Core logic that ties the functionalities from PaLM to LangChain.

Used internally by the Google Generative AI vector store.
End users should not use this directly.

Instead, use the public API:

    langchain.vectorstores.google.generativeai.GoogleVectorStore.
"""

import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import google.ai.generativelanguage as genai
from langchain.pydantic_v1 import BaseModel, PrivateAttr

from . import _genai_extension as genaix
from .semantic_retriever import DoesNotExistsException

VST = TypeVar("VST", bound="SemanticRetriever")
logger = logging.getLogger(__name__)


class SemanticRetriever(BaseModel):
    name: genaix.EntityName
    _client: genai.RetrieverServiceClient = PrivateAttr()

    def __init__(self, *, client: genai.RetrieverServiceClient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    def from_ids(
        cls, corpus_id: str, document_id: Optional[str]
    ) -> "SemanticRetriever":
        name = genaix.EntityName(corpus_id=corpus_id, document_id=document_id)
        client = genaix.build_semantic_retriever()

        # Check the entity exists on Google server.
        if name.is_corpus():
            if genaix.get_corpus(corpus_id=corpus_id, client=client) is None:
                raise DoesNotExistsException(corpus_id=corpus_id)
        elif name.is_document():
            assert document_id is not None
            if (
                genaix.get_document(
                    corpus_id=corpus_id, document_id=document_id, client=client
                )
                is None
            ):
                raise DoesNotExistsException(
                    corpus_id=corpus_id, document_id=document_id
                )

        return cls(name=name, client=client)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None,
    ) -> List[str]:
        if self.name.document_id is None and document_id is None:
            raise NotImplementedError(
                "Adding texts to a corpus directly is not supported. "
                "Please provide a document ID under the corpus first. "
                "Then add the texts to the document."
            )
        if (
            self.name.document_id is not None
            and document_id is not None
            and self.name.document_id != document_id
        ):
            raise NotImplementedError(
                f"Parameter `document_id` {document_id} does not match the "
                f"vector store's `document_id` {self.name.document_id}"
            )
        assert self.name.document_id or document_id is not None
        new_document_id = self.name.document_id or document_id or ""

        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != len(metadatas):
            raise ValueError(
                f"metadatas's length {len(metadatas)} and "
                f"texts's length {len(texts)} are mismatched"
            )

        chunks = genaix.batch_create_chunk(
            corpus_id=self.name.corpus_id,
            document_id=new_document_id,
            texts=texts,
            metadatas=metadatas,
            client=self._client,
        )

        return [chunk.name for chunk in chunks if chunk.name]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if self.name.is_corpus():
            relevant_chunks = genaix.query_corpus(
                corpus_id=self.name.corpus_id,
                query=query,
                k=k,
                filter=filter,
                client=self._client,
            )
        else:
            assert self.name.is_document()
            assert self.name.document_id is not None
            relevant_chunks = genaix.query_document(
                corpus_id=self.name.corpus_id,
                document_id=self.name.document_id,
                query=query,
                k=k,
                filter=filter,
                client=self._client,
            )

        return [
            (chunk.chunk.data.string_value, chunk.chunk_relevance_score)
            for chunk in relevant_chunks
        ]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        for id in ids or []:
            name = genaix.EntityName.from_str(id)
            _delete(
                corpus_id=name.corpus_id,
                document_id=name.document_id,
                chunk_id=name.chunk_id,
                client=self._client,
            )
        return True


def _delete(
    *,
    corpus_id: str,
    document_id: Optional[str],
    chunk_id: Optional[str],
    client: genai.RetrieverServiceClient,
) -> None:
    if chunk_id is not None:
        if document_id is None:
            raise ValueError(f"Chunk {chunk_id} requires a document ID")
        genaix.delete_chunk(
            corpus_id=corpus_id,
            document_id=document_id,
            chunk_id=chunk_id,
            client=client,
        )
    elif document_id is not None:
        genaix.delete_document(
            corpus_id=corpus_id, document_id=document_id, client=client
        )
    else:
        genaix.delete_corpus(corpus_id=corpus_id, client=client)
