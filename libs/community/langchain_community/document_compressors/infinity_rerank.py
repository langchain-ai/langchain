from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from pydantic import ConfigDict, model_validator

if TYPE_CHECKING:
    from infinity_client.api.default import rerank
    from infinity_client.client import Client
    from infinity_client.models import RerankInput
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from infinity_client.api.default import rerank
        from infinity_client.client import Client
        from infinity_client.models import RerankInput
    except ImportError:
        pass

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_BASE_URL = "http://localhost:7997"


class InfinityRerank(BaseDocumentCompressor):
    """Document compressor that uses `Infinity Rerank API`."""

    client: Optional[Client] = None
    """Infinity client to use for compressing documents."""

    model: Optional[str] = None
    """Model to use for reranking."""

    top_n: Optional[int] = 3
    """Number of documents to return."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that python package exists in environment."""
        if "client" in values:
            return values
        else:
            try:
                from infinity_client.client import Client
            except ImportError:
                raise ImportError(
                    "Could not import infinity_client python package. "
                    "Please install it with `pip install infinity_client`."
                )

            values["model"] = values.get("model", DEFAULT_MODEL_NAME)
            values["client"] = Client(base_url=DEFAULT_BASE_URL)
            return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
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

        input = RerankInput(
            query=query,
            documents=docs,
            model=model,
        )
        results = rerank.sync(client=self.client, body=input)

        if hasattr(results, "results"):
            results = getattr(results, "results")

        result_dicts = []
        for res in results:
            result_dicts.append(
                {
                    "index": res.index,
                    "relevance_score": res.relevance_score,
                }
            )

        result_dicts.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        return result_dicts[:top_n]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Infinity's rerank API.

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
