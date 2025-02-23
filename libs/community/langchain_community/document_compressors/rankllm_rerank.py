from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from pydantic import ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from rank_llm.data import Candidate, Query, Request
    from rank_llm.rerank.reranker import Reranker
else:
    try:
        from rank_llm.data import Candidate, Query, Request
        from rank_llm.rerank.reranker import Reranker
    except ImportError:
        pass


class RankLLMRerank(BaseDocumentCompressor):
    """Document compressor using RankLLM interface."""

    client: Any = None
    """RankLLM client to use for compressing documents"""
    top_n: Optional[int] = Field(default=3)
    """Top N documents to return."""
    model: str = Field(default="rank_zephyr")
    """Model name to use for reranking."""
    window_size: Optional[int] = Field(default=None)
    """Window size for reranking (listwise models only)."""
    batch_size: Optional[int] = Field(default=None)
    """Batch size for reranking (pointwise models only)."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate RankLLM installation and initialize the reranker."""

        if not values.get("client"):
            model_name = values.get("model", "rank_zephyr").lower()

            try:
                values["client"] = Reranker.create_agent(
                    model_name,
                    default_agent=None,
                    interactive=False,
                    window_size=values.get("window_size"),
                    batch_size=values.get("batch_size"),
                )
            except ImportError:
                raise ImportError(
                    "Could not import RankLLM. Install it with `pip install rank-llm`."
                )

        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using RankLLM reranking."""

        request: List[Request] = [
            Request(
                query=Query(text=query, qid=1),
                candidates=[
                    Candidate(
                        docid=index,
                        score=1,
                        doc={
                            "body": doc.page_content,
                            "headings": "",
                            "title": "",
                            "url": "",
                        },
                    )
                    for index, doc in enumerate(documents)
                ],
            )
        ]

        rerank_results = self.client.rerank_batch(
            request,
            rank_end=len(request[0].candidates),
            rank_start=0,
            shuffle_candidates=False,
            logging=False,
            top_k_retrieve=len(request[0].candidates),
        )

        final_results = []
        for candidate in rerank_results[0].candidates:
            doc = documents[int(candidate.docid)]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            final_results.append(doc_copy)

        return final_results[: self.top_n] if self.top_n is not None else final_results
