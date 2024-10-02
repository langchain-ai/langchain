from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, model_validator

JINA_API_URL: str = "https://api.jina.ai/v1/rerank"


class JinaRerank(BaseDocumentCompressor):
    """Document compressor that uses `Jina Rerank API`."""

    session: Any = None
    """Requests session to communicate with API."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "jina-reranker-v1-base-en"
    """Model to use for reranking."""
    jina_api_key: Optional[str] = None
    """Jina API key. Must be specified directly or via environment variable 
        JINA_API_KEY."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        jina_api_key = get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
        user_agent = values.get("user_agent", "langchain")
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
                "user-agent": user_agent,
            }
        )
        values["session"] = session
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
        max_chunks_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            max_chunks_per_doc : The maximum number of chunks derived from a document.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        data = {
            "query": query,
            "documents": docs,
            "model": model,
            "top_n": top_n,
        }

        resp = self.session.post(
            JINA_API_URL,
            json=data,
        ).json()

        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["results"]
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
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Jina's Rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

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
