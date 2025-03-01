from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from pinecone import Pinecone
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class PineconeRerank(BaseDocumentCompressor):
    """Document compressor that uses `Pinecone Rerank API`."""

    client: Any = None
    """Pinecone client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: Optional[str] = None
    """Model to use for reranking. Mandatory to specify the model name."""
    pinecone_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("PINECONE_API_KEY", default=None)
    )
    """Pinecone API key. Must be specified directly or via environment variable 
    PINECONE_API_KEY."""
    rank_fields: Optional[Sequence[str]] = None
    """Fields to use for reranking when documents are dictionaries."""
    return_documents: bool = True
    """Whether to return the documents in the reranking results."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:  # type: ignore[valid-type]
        """Validate that api key and python package exists in environment."""
        if not self.client:
            if isinstance(self.pinecone_api_key, SecretStr):
                pinecone_api_key: Optional[str] = self.pinecone_api_key.get_secret_value()
            else:
                pinecone_api_key = self.pinecone_api_key
            self.client = Pinecone(api_key=pinecone_api_key)
        elif not isinstance(self.client, Pinecone):
            raise ValueError(
                "The 'client' parameter must be an instance of pinecone.Pinecone.\n"
                "You may create the Pinecone object like:\n\n"
                "from pinecone import Pinecone\nclient = Pinecone(api_key=...)"
            )
        return self

    @model_validator(mode="after")
    def validate_model_specified(self) -> Self:  # type: ignore[valid-type]
        """Validate that model is specified."""
        if not self.model:
            raise ValueError(
                "Did not find `model`! Please "
                " pass `model` as a named parameter."
                " Example models include 'bge-reranker-v2-m3'."
            )

        return self

    def _document_to_dict(
        self,
        document: Union[str, Document, dict],
        index: int,
    ) -> dict:
        if isinstance(document, Document):
            return {"id": f"doc{index}", "content": document.page_content}
        elif isinstance(document, dict):
            if not document.get("id"):
                document["id"] = f"doc{index}"
            return document
        else:
            return {"id": f"doc{index}", "content": document}

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        truncate: str = "END",
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            rank_fields: A sequence of keys to use for reranking.
            top_n: The number of results to return. If None returns all results.
                Defaults to self.top_n.
            truncate: How to truncate documents if they exceed token limits. Options: "END", 
                "MIDDLE". Defaults to "END".
        """
        if len(documents) == 0:  # to avoid empty API call
            return []
            
        docs = [self._document_to_dict(doc, i) for i, doc in enumerate(documents)]
        model = model or self.model
        rank_fields = rank_fields or self.rank_fields
        top_n = top_n if top_n is not None else self.top_n
        
        parameters = {"truncate": truncate}
        
        results = self.client.inference.rerank(
            model=model,
            query=query,
            documents=docs,
            rank_fields=rank_fields,
            top_n=top_n,
            return_documents=self.return_documents,
            parameters=parameters
        )
        
        result_dicts = []
        for res in results:
            result_dict = {
                "id": res.id,
                "index": next((i for i, doc in enumerate(docs) if doc["id"] == res.id), None),
                "score": res.score
            }
            if hasattr(res, "document") and res.document:
                result_dict["document"] = res.document
            result_dicts.append(result_dict)
            
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Pinecone's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        reranked_results = self.rerank(documents, query)
        
        for res in reranked_results:
            if res["index"] is not None:
                doc = documents[res["index"]]
                doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                doc_copy.metadata["relevance_score"] = res["score"]
                compressed.append(doc_copy)
        
        return compressed
