"""Semantic similarity-based text splitter."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from langchain_text_splitters.base import TextSplitter

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1, 1].
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences using a regex heuristic.

    Args:
        text: The raw input text to split into sentences.

    Returns:
        A list of non-empty sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace/end-of-string.
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


class SemanticSimilarityTextSplitter(TextSplitter):
    """Split text into chunks based on semantic similarity of adjacent sentences.

    Uses an `Embeddings` model to embed sentences and calculates cosine similarity
    between consecutive sentences. Breakpoints are inserted where a similarity "valley"
    is detected, keeping semantically related content together.

    This is especially useful for RAG (Retrieval-Augmented Generation) pipelines
    where coherent, topic-focused chunks improve retrieval precision.

    !!! note
        This splitter ignores `chunk_size` and `chunk_overlap` parameters inherited
        from `TextSplitter`. Chunk boundaries are determined purely by semantic
        breakpoints derived from the `breakpoint_threshold_type` strategy.

    Example:
        .. code-block:: python

            from langchain_text_splitters.semantic import SemanticSimilarityTextSplitter
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            splitter = SemanticSimilarityTextSplitter(embeddings=embeddings)
            chunks = splitter.split_text("Your long document text here...")
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile"
        ] = "percentile",
        breakpoint_threshold_amount: float | None = None,
        sentence_split_regex: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new `SemanticSimilarityTextSplitter`.

        Args:
            embeddings: A `langchain_core.embeddings.Embeddings` instance used to
                embed sentences. Any compatible provider (OpenAI, HuggingFace, Ollama,
                etc.) can be used.
            breakpoint_threshold_type: Strategy to determine breakpoint thresholds
                from the distribution of cosine similarity scores.

                - `"percentile"`: Break where similarity drops below the Nth
                  percentile. Default `breakpoint_threshold_amount` is `95`.
                - `"standard_deviation"`: Break where similarity drops more than N
                  standard deviations below the mean. Default is `3`.
                - `"interquartile"`: Break where similarity drops below
                  Q1 - N * IQR. Default is `1.5`.
            breakpoint_threshold_amount: Numeric threshold for the chosen strategy.
                If `None`, a sensible default for each strategy is used.
            sentence_split_regex: Custom regex pattern used to split text into
                sentences before embedding. If `None`, a built-in heuristic
                (split on sentence-ending punctuation) is used.
            **kwargs: Additional keyword arguments forwarded to `TextSplitter`.

        Raises:
            ValueError: If `breakpoint_threshold_type` is not one of the supported
                values.
            ValueError: If provided `breakpoint_threshold_amount` is negative.
        """
        super().__init__(**kwargs)
        self._embeddings = embeddings
        self._breakpoint_threshold_type = breakpoint_threshold_type
        self._sentence_split_regex = sentence_split_regex

        _defaults: dict[str, float] = {
            "percentile": 95.0,
            "standard_deviation": 3.0,
            "interquartile": 1.5,
        }
        if breakpoint_threshold_type not in _defaults:
            msg = (
                f"Invalid breakpoint_threshold_type '{breakpoint_threshold_type}'. "
                f"Must be one of: {list(_defaults.keys())}"
            )
            raise ValueError(msg)

        resolved_amount = (
            breakpoint_threshold_amount
            if breakpoint_threshold_amount is not None
            else _defaults[breakpoint_threshold_type]
        )
        if resolved_amount < 0:
            msg = (
                f"breakpoint_threshold_amount must be >= 0, got {resolved_amount}"
            )
            raise ValueError(msg)
        self._breakpoint_threshold_amount = resolved_amount

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Split the input text into semantically coherent chunks.

        Steps:
        1. Sentence-split the text.
        2. Embed each sentence.
        3. Compute cosine similarity between adjacent sentences.
        4. Identify breakpoints where similarity falls below the threshold.
        5. Aggregate sentences into final chunks.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks, each representing a coherent semantic unit.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self._embeddings.embed_documents(sentences)
        similarities = self._compute_similarities(embeddings)
        breakpoint_threshold = self._calculate_threshold(similarities)
        breakpoints = self._find_breakpoints(similarities, breakpoint_threshold)
        return self._build_chunks(sentences, breakpoints)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using the configured regex or built-in heuristic.

        Args:
            text: Raw input text.

        Returns:
            A list of sentence strings.
        """
        if self._sentence_split_regex:
            pattern = re.compile(self._sentence_split_regex)
            parts = pattern.split(text.strip())
            return [s.strip() for s in parts if s.strip()]
        return _split_sentences(text)

    def _compute_similarities(
        self, embeddings: list[list[float]]
    ) -> list[float]:
        """Compute cosine similarity between each pair of adjacent sentence embeddings.

        Args:
            embeddings: A list of embedding vectors, one per sentence.

        Returns:
            A list of similarity scores of length `len(embeddings) - 1`.
        """
        return [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

    def _calculate_threshold(self, similarities: list[float]) -> float:
        """Calculate the breakpoint similarity threshold from the distribution.

        Args:
            similarities: List of cosine similarity scores between adjacent sentences.

        Returns:
            The numeric threshold below which a breakpoint is placed.
        """
        n = len(similarities)
        if n == 0:
            return 0.0

        sorted_sims = sorted(similarities)

        if self._breakpoint_threshold_type == "percentile":
            # Lower percentile -> more splits; N=95 keeps only clear breaks.
            # We invert: threshold is the (100 - amount)th percentile.
            pct = 100.0 - self._breakpoint_threshold_amount
            idx = int(pct / 100.0 * (n - 1))
            return sorted_sims[max(0, min(idx, n - 1))]

        if self._breakpoint_threshold_type == "standard_deviation":
            mean = sum(similarities) / n
            variance = sum((s - mean) ** 2 for s in similarities) / n
            std = variance ** 0.5
            return mean - self._breakpoint_threshold_amount * std

        # interquartile
        q1_idx = int(0.25 * (n - 1))
        q3_idx = int(0.75 * (n - 1))
        q1 = sorted_sims[q1_idx]
        q3 = sorted_sims[q3_idx]
        iqr = q3 - q1
        return q1 - self._breakpoint_threshold_amount * iqr

    def _find_breakpoints(
        self, similarities: list[float], threshold: float
    ) -> list[int]:
        """Find sentence indices after which a chunk break should occur.

        A break is placed after sentence `i` when the similarity between sentence `i`
        and sentence `i+1` covers or falls below `threshold`.

        Args:
            similarities: List of per-adjacent-pair cosine similarities.
            threshold: The maximum similarity score to trigger a split.

        Returns:
            Sorted list of sentence indices after which a new chunk begins.
        """
        return [i for i, sim in enumerate(similarities) if sim <= threshold]

    def _build_chunks(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[str]:
        """Aggregate sentences into chunks using the identified breakpoints.

        Args:
            sentences: The original list of sentence strings.
            breakpoints: Indices after which a new chunk starts.

        Returns:
            A list of chunk strings formed by joining sentences within each segment.
        """
        chunks: list[str] = []
        start = 0
        for bp in breakpoints:
            chunk = " ".join(sentences[start : bp + 1])
            if chunk:
                chunks.append(chunk)
            start = bp + 1
        # Append remaining sentences as the final chunk.
        final_chunk = " ".join(sentences[start:])
        if final_chunk:
            chunks.append(final_chunk)
        return chunks
