from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.pydantic_v1 import Extra
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.schema.cross_encoder import CrossEncoder


class CrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: CrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """

        pairs = list(map(lambda doc: [query, doc.page_content], documents))
        scores = self.model.score(pairs)

        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        reranked_docs = list(map(lambda x: x[0], result))
        scores = list(map(lambda x: x[1], result))

        final_results = []
        for index in range(self.top_n):
            if len(reranked_docs) <= index:
                break
            doc = reranked_docs[index]
            doc.metadata["relevance_score"] = scores[index]
            final_results.append(doc)

        return final_results
