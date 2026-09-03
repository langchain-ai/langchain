"""Lexical similarity document chunk deduplicator."""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

try:
    from langchain_core.documents import BaseDocumentTransformer, Document
except ImportError:
    try:
        from langchain_core.documents.base import Document
        from langchain_core.documents.transformers import BaseDocumentTransformer
    except ImportError:
        class Document:  # type: ignore[no-redef]
            def __init__(
                self,
                page_content: str,
                metadata: Optional[dict[str, Any]] = None,
            ) -> None:
                self.page_content = page_content
                self.metadata = metadata or {}

        from abc import ABC
        class BaseDocumentTransformer(ABC):  # type: ignore[no-redef]
            pass


class LexicalSimilarityDeduplicator(BaseDocumentTransformer, BaseModel):
    """Document transformer that deduplicates document chunks using lexical Jaccard similarity.

    Filters out near-duplicate and highly overlapping document chunks retrieved
    during RAG pipelines to prevent context window saturation and reduce token costs.

    Example:
        .. code-block:: python

            from langchain_community.document_transformers import LexicalSimilarityDeduplicator
            from langchain_core.documents import Document

            docs = [
                Document(page_content="LangChain is a framework for developing applications powered by LLMs."),
                Document(page_content="LangChain is a framework for developing applications powered by LLMs."),
                Document(page_content="Python is a popular programming language.")
            ]

            deduplicator = LexicalSimilarityDeduplicator(similarity_threshold=0.8, ngram_size=2)
            unique_docs = deduplicator.transform_documents(docs)
    """

    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold above which chunks are considered duplicates.",
    )
    ngram_size: int = Field(
        default=2,
        ge=1,
        description="Size of word n-grams used for lexical representation.",
    )

    class Config:
        arbitrary_types_allowed = True

    def _tokenize_to_ngrams(self, text: str) -> Set[str]:
        """Normalize text and extract word n-grams.

        Args:
            text: Input string to tokenize.

        Returns:
            Set of extracted n-gram strings.
        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
        words = cleaned.split()

        if not words:
            return set()

        if len(words) < self.ngram_size:
            return {cleaned}

        ngrams: Set[str] = set()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i : i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def _calculate_jaccard_similarity(
        self, set_a: Set[str], set_b: Set[str]
    ) -> float:
        """Calculate Jaccard similarity coefficient between two sets.

        Args:
            set_a: First set of n-grams.
            set_b: Second set of n-grams.

        Returns:
            Jaccard similarity index between 0.0 and 1.0.
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union > 0 else 0.0

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Filter out duplicate document chunks based on lexical Jaccard similarity.

        Args:
            documents: Sequence of Document objects to deduplicate.

        Returns:
            Sequence of deduplicated Document objects preserving relative order.
        """
        unique_documents: List[Document] = []
        accepted_ngrams_list: List[Set[str]] = []

        for doc in documents:
            doc_ngrams = self._tokenize_to_ngrams(doc.page_content)
            is_duplicate = False

            for accepted_ngrams in accepted_ngrams_list:
                sim = self._calculate_jaccard_similarity(doc_ngrams, accepted_ngrams)
                if sim >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_documents.append(doc)
                accepted_ngrams_list.append(doc_ngrams)

        return unique_documents

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously filter out duplicate document chunks.

        Args:
            documents: Sequence of Document objects to deduplicate.

        Returns:
            Sequence of deduplicated Document objects.
        """
        return self.transform_documents(documents, **kwargs)
