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
        Similarity score in range [-1, 1].
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
        text: The raw input text.

    Returns:
        A list of non-empty sentence strings.
    """
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

    This splitter respects the `chunk_size` parameter. If a semantically
    identified chunk exceeds `chunk_size`, the splitter hierarchically finds
    the next most significant semantic breakpoints within that chunk until
    the size constraint is met or no further splits (sentences) are available.

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
        """Create a new semantic similarity text splitter.

        Args:
            embeddings: Embedding model used to vectorize sentences.
            breakpoint_threshold_type: Strategy to determine breakpoint thresholds
                from the distribution of cosine similarity scores. Choices include
                percentile, standard_deviation, and interquartile.
            breakpoint_threshold_amount: Numeric threshold for the chosen strategy.
                If None, a built-in default for each strategy is applied.
            sentence_split_regex: Custom regex pattern used to split text into
                sentences before embedding. If None, uses a punctuation-based heuristic.
            **kwargs: Additional keyword arguments forwarded to the base class.

        Raises:
            ValueError: If an unsupported threshold type is provided.
            ValueError: If the threshold amount is negative.
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


    def split_text(self, text: str) -> list[str]:
        """Split the input text into semantically coherent chunks.

        Steps include sentence-splitting the text, embedding each sentence,
        and computing similarities. Large chunks are refined recursively.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks that respect the size constraint.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self._embeddings.embed_documents(sentences)
        similarities = self._compute_similarities(embeddings)

        breakpoint_threshold = self._calculate_threshold(similarities)
        breakpoints = set(self._find_breakpoints(similarities, breakpoint_threshold))

        final_breakpoints = self._hierarchical_refine(
            sentences, similarities, breakpoints
        )

        return self._build_chunks(sentences, final_breakpoints)


    def _hierarchical_refine(
        self,
        sentences: list[str],
        similarities: list[float],
        breakpoints: set[int],
    ) -> list[int]:
        """Recursively refine breakpoints until chunks meet the size limit.

        Args:
            sentences: List of original sentences.
            similarities: Adjacent sentence similarities.
            breakpoints: Current set of split indices.

        Returns:
            Sorted list of all breakpoints.
        """
        while True:
            sorted_breakpoints = sorted(list(breakpoints))
            chunk_ranges = []
            start = 0
            for breakpoint_idx in sorted_breakpoints:
                chunk_ranges.append((start, breakpoint_idx))
                start = breakpoint_idx + 1
            chunk_ranges.append((start, len(sentences) - 1))

            split_occurred = False
            for start_idx, end_idx in chunk_ranges:
                chunk_text = " ".join(sentences[start_idx : end_idx + 1])
                if (
                    len(chunk_text) > self._chunk_size
                    and (end_idx - start_idx) > 0
                ):
                    sub_similarities = similarities[start_idx:end_idx]
                    if not sub_similarities:
                        continue

                    min_sim = min(sub_similarities)
                    sub_breakpoint_idx = start_idx + sub_similarities.index(min_sim)

                    if sub_breakpoint_idx not in breakpoints:
                        breakpoints.add(sub_breakpoint_idx)
                        split_occurred = True
                        break

            if not split_occurred:
                break

        return sorted(list(breakpoints))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex or built-in heuristic.

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
        """Compute cosine similarity between each adjacent sentence pair.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Similarity scores of length n-1.
        """
        return [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

    def _calculate_threshold(self, similarities: list[float]) -> float:
        """Calculate the breakpoint threshold from the distribution.

        Args:
            similarities: List of cosine similarity scores.

        Returns:
            The threshold below which a breakpoint is placed.
        """
        n = len(similarities)
        if n == 0:
            return 0.0

        sorted_sims = sorted(similarities)

        if self._breakpoint_threshold_type == "percentile":
            pct = 100.0 - self._breakpoint_threshold_amount
            idx = int(pct / 100.0 * (n - 1))
            return sorted_sims[max(0, min(idx, n - 1))]

        if self._breakpoint_threshold_type == "standard_deviation":
            mean = sum(similarities) / n
            variance = sum((s - mean) ** 2 for s in similarities) / n
            std = variance ** 0.5
            return mean - self._breakpoint_threshold_amount * std

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

        A break is placed after sentence `i` when the similarity between `i`
        and `i+1` falls below the threshold. Ensures the threshold is strictly
        less than the maximum similarity in the document.

        Args:
            similarities: List of adjacent-pair similarities.
            threshold: The maximum similarity score to trigger a split.

        Returns:
            Sentence indices after which a new chunk begins.
        """
        unique_sims = sorted(list(set(similarities)))
        if len(unique_sims) <= 1:
            return []

        max_sim = unique_sims[-1]
        if threshold >= max_sim:
            threshold = unique_sims[-2]

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
        for breakpoint_idx in breakpoints:
            chunk = " ".join(sentences[start : breakpoint_idx + 1])
            if chunk:
                chunks.append(chunk)
            start = breakpoint_idx + 1
        final_chunk = " ".join(sentences[start:])
        if final_chunk:
            chunks.append(final_chunk)
        return chunks
