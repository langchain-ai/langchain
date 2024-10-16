from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, model_validator

if TYPE_CHECKING:
    from flashrank import Ranker, RerankRequest
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        pass

DEFAULT_MODEL_NAME = "ms-marco-MultiBERT-L-12"


class FlashrankRerank(BaseDocumentCompressor):
    """Document compressor using Flashrank interface."""

    client: Ranker
    """Flashrank client to use for compressing documents"""
    top_n: int = 3
    """Number of documents to return."""
    score_threshold: float = 0.0
    """Minimum relevance threshold to return."""
    model: Optional[str] = None
    """Model to use for reranking."""
    prefix_metadata: str = ""
    """Prefix for flashrank_rerank metadata keys"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        if "client" in values:
            return values
        else:
            try:
                from flashrank import Ranker
            except ImportError:
                raise ImportError(
                    "Could not import flashrank python package. "
                    "Please install it with `pip install flashrank`."
                )

            values["model"] = values.get("model", DEFAULT_MODEL_NAME)
            values["client"] = Ranker(model_name=values["model"])
            return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_response = self.client.rerank(rerank_request)[: self.top_n]
        final_results = []

        for r in rerank_response:
            if r["score"] >= self.score_threshold:
                doc = Document(
                    page_content=r["text"],
                    metadata={
                        self.prefix_metadata + "id": r["id"],
                        self.prefix_metadata + "relevance_score": r["score"],
                        **r["meta"],
                    },
                )
                final_results.append(doc)
        return final_results
