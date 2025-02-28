from __future__ import annotations

from copy import deepcopy
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Union
)
from langchain_core.documents import BaseDocumentCompressor, Document


class XinferenceRerank(BaseDocumentCompressor):
    """Document compressor that uses `Xinference Rerank API`."""

    top_n: Optional[int] = 3
    client: Any
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""

    def __init__(
        self, 
        server_url: Optional[str] = None, 
        model_uid: Optional[str] = None, 
        top_n: Optional[int] = 3
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            try:
                from xinference_client import RESTfulClient
            except ImportError as e:
                raise ImportError(
                    "Could not import RESTfulClient from xinference. Please install it"
                    " with `pip install xinference` or `pip install xinference_client`."
                ) from e

        super().__init__()

        if server_url is None:
            raise ValueError("Please provide server URL")

        if model_uid is None:
            raise ValueError("Please provide the model UID")

        self.server_url = server_url
        self.model_uid = model_uid
        self.top_n = top_n
        self.client = RESTfulClient(server_url)


    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents \
        ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            max_chunks_per_doc : The maximum number of chunks derived from a document.
        """  
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = self.client.get_model(self.model_uid)
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = model.rerank(
            documents=docs,
            query=query,
            top_n=top_n,
            max_chunks_per_doc=max_chunks_per_doc,
            return_documents=return_documents,
            return_len=return_len
        )
        if hasattr(results, "results"):
            results = getattr(results, "results")
        results = results["results"]
        result_dicts = []
        for res in results:
            result_dicts.append(
                {"index": res["index"], "relevance_score": res["relevance_score"]}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """
        Compress documents using Xinference's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
