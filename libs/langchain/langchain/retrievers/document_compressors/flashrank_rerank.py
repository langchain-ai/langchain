from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

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
    model: Optional[str] = None
    """Model to use for reranking."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
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
            metadata = r["meta"]
            metadata["relevance_score"] = r["score"]
            doc = Document(
                page_content=r["text"],
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results
