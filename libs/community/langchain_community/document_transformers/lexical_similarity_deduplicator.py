"""
Lexical Similarity Chunk Deduplicator for LangChain
Filters out redundant, highly overlapping document chunks in RAG retrieval pipelines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set


class Document:
    """Standard LangChain Document representation."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:30]}...', metadata={self.metadata})"


class LexicalSimilarityDeduplicator:
    """
    Document transformer that filters near-duplicate document chunks from RAG context
    using n-gram Jaccard lexical similarity.

    Prevents context window pollution and reduces LLM inference token cost.

    Parameters:
        similarity_threshold: Float between 0.0 and 1.0 (default 0.75).
                              Documents with similarity >= threshold against already
                              accepted documents are discarded as redundant.
        ngram_size: Size of word n-grams for Jaccard calculation (default 2 for bigrams).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        ngram_size: int = 2,
    ) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")

        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    def _tokenize_to_ngrams(self, text: str) -> Set[str]:
        """Normalizes text and extracts word n-grams."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
        words = cleaned.split()

        if not words:
            return set()

        if len(words) < self.ngram_size:
            return {cleaned}

        ngrams = set()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i : i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def _calculate_jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Calculates Jaccard similarity index between two n-gram sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union > 0 else 0.0

    def transform_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Filters and returns a deduplicated list of documents.
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

    async def atransform_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Asynchronous document transformation wrapper."""
        return self.transform_documents(documents)
