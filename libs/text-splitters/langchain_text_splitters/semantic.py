"""Semantic text splitter based on embedding similarity.

Splits text at natural thematic boundaries by measuring cosine distance between
consecutive sentence embeddings, rather than splitting at fixed character counts.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from langchain_core.embeddings import Embeddings

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


logger = logging.getLogger(__name__)

BreakpointThresholdType = Literal[
    "percentile",
    "standard_deviation",
    "interquartile",
    "gradient",
]

_BREAKPOINT_DEFAULTS: dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}


class _SentenceGroup(NamedTuple):
    """Internal representation of a sentence with its positional context."""

    sentence: str
    sentence_index: int
    combined_sentence: str


class _SentenceSpan(NamedTuple):
    """Internal representation of a sentence span in the original text."""

    sentence: str
    sentence_index: int
    start_index: int
    end_index: int


class _ChunkRange(NamedTuple):
    """Internal start/end offsets for a chunk in the original text."""

    start_index: int
    end_index: int


def _combine_sentences(
    sentences: list[str], *, buffer_size: int = 1
) -> list[_SentenceGroup]:
    """Build context windows around each sentence by including neighbors.

    Each sentence is combined with its surrounding sentences (determined by
    ``buffer_size``) to produce a richer text for embedding. This helps the
    embedding model capture cross-sentence semantic relationships.

    Args:
        sentences: Individual sentences extracted from the source text.
        buffer_size: Number of neighboring sentences on each side to include
            in the combined representation.

    Returns:
        A list of ``_SentenceGroup`` named tuples, each containing the original
        sentence, its index, and the combined context string.
    """
    groups: list[_SentenceGroup] = []
    for i, sentence in enumerate(sentences):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        combined = " ".join(sentences[start:end])
        groups.append(
            _SentenceGroup(
                sentence=sentence, sentence_index=i, combined_sentence=combined
            )
        )
    return groups


def _cosine_distances_between_consecutive(
    embeddings: list[list[float]],
) -> list[float]:
    """Compute cosine distances between each pair of consecutive embeddings.

    Uses the formula: ``distance = 1 - cosine_similarity``, where cosine
    similarity is computed via the dot product of L2-normalized vectors.

    Args:
        embeddings: A list of embedding vectors, one per sentence group.

    Returns:
        A list of ``len(embeddings) - 1`` cosine distances. The ``i``-th
        element is the distance between ``embeddings[i]`` and
        ``embeddings[i + 1]``.
    """
    embedding_array = np.array(embeddings, dtype=np.float64)

    # L2-normalize each row; handle zero-norm rows gracefully.
    norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embedding_array / norms

    # Cosine similarity between consecutive pairs via row-wise dot product.
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    # Clip to [-1, 1] to guard against floating-point drift.
    similarities = np.clip(similarities, -1.0, 1.0)

    distances: list[float] = (1.0 - similarities).tolist()
    return distances


def _merge_small_chunks(chunks: list[str], *, min_chunk_size: int) -> list[str]:
    """Merge chunks that are shorter than ``min_chunk_size`` into neighbors.

    Small chunks are folded into the *previous* chunk when possible, or into
    the *next* chunk when the small chunk is the first in the list. This
    preserves all content rather than dropping undersized chunks.

    Args:
        chunks: The initial list of text chunks to consolidate.
        min_chunk_size: Minimum character length for a chunk to stand alone.

    Returns:
        A new list of chunks where no chunk is shorter than ``min_chunk_size``
        (unless the entire input text is shorter).
    """
    if not chunks:
        return []

    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) < min_chunk_size:
            # Previous chunk is too small — absorb the current one into it.
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Final pass: if the last chunk is still too small, merge it backward.
    if len(merged) > 1 and len(merged[-1]) < min_chunk_size:
        last = merged.pop()
        merged[-1] = merged[-1] + " " + last

    return merged


class SemanticChunker(BaseDocumentTransformer):
    """Split text into semantically coherent chunks using embedding similarity.

    Unlike character- or token-based splitters, ``SemanticChunker`` identifies
    natural thematic boundaries by:

    1. Splitting the text into sentences.
    2. Building context windows around each sentence (controlled by
       ``buffer_size``).
    3. Embedding the context windows using the provided ``Embeddings`` model.
    4. Computing cosine distances between consecutive embeddings.
    5. Placing chunk boundaries where the distance exceeds a threshold.

    Four threshold strategies are available (``breakpoint_threshold_type``):

    * ``"percentile"`` — split where the distance exceeds the *N*-th
      percentile of all distances.
    * ``"standard_deviation"`` — split where the distance exceeds
      ``mean + N * std``.
    * ``"interquartile"`` — split where the distance exceeds
      ``mean + N * IQR``.
    * ``"gradient"`` — split where the *rate of change* of distances
      exceeds the *N*-th percentile.

    Alternatively, set ``number_of_chunks`` to let the chunker automatically
    find the threshold that produces the desired number of chunks.

    !!! warning

        This class requires ``numpy`` to be installed. Install it with
        ``pip install numpy``.

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import SemanticChunker

        chunker = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        chunks = chunker.split_text("Your long document text here...")
        ```

    Args:
        embeddings: The embedding model used to compute sentence vectors.
        buffer_size: Number of neighboring sentences on each side to include
            when building the context window for embedding.
        add_start_index: If ``True``, each ``Document`` produced by
            ``create_documents`` will include a ``start_index`` key in its
            metadata indicating the character offset in the original text.
        breakpoint_threshold_type: Strategy for determining chunk boundaries.
        breakpoint_threshold_amount: Numeric parameter for the chosen
            threshold strategy. When ``None``, a sensible default is used
            (95 for percentile, 3 for standard_deviation, 1.5 for
            interquartile, 95 for gradient).
        number_of_chunks: If set, the chunker finds the threshold that
            produces this many chunks. If ``breakpoint_threshold_amount``
            is also provided, ``number_of_chunks`` takes precedence for
            backward compatibility with the experimental implementation.
        sentence_split_regex: Regular expression used to split the input
            text into sentences.
        min_chunk_size: Minimum character length for a chunk. Chunks shorter
            than this are merged into a neighbor rather than discarded.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,  # noqa: FBT001, FBT002
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: float | None = None,
        number_of_chunks: int | None = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: int | None = None,
    ) -> None:
        """Initialize the ``SemanticChunker``.

        Raises:
            ImportError: If numpy is not installed.
            ValueError: If ``breakpoint_threshold_type`` is unknown.
            ValueError: If ``buffer_size`` is negative.
            ValueError: If ``number_of_chunks`` is less than 1.
            ValueError: If ``min_chunk_size`` is less than 1.
        """
        if not _HAS_NUMPY:
            msg = "SemanticChunker requires numpy. Install it with `pip install numpy`."
            raise ImportError(msg)

        if breakpoint_threshold_type not in _BREAKPOINT_DEFAULTS:
            msg = (
                "Got unexpected `breakpoint_threshold_type`: "
                f"{breakpoint_threshold_type}"
            )
            raise ValueError(msg)

        if buffer_size < 0:
            msg = f"`buffer_size` must be >= 0, got {buffer_size}"
            raise ValueError(msg)

        if number_of_chunks is not None and number_of_chunks < 1:
            msg = f"`number_of_chunks` must be >= 1, got {number_of_chunks}"
            raise ValueError(msg)

        if min_chunk_size is not None and min_chunk_size < 1:
            msg = f"`min_chunk_size` must be >= 1, got {min_chunk_size}"
            raise ValueError(msg)

        if number_of_chunks is not None and breakpoint_threshold_amount is not None:
            logger.warning(
                "Both `number_of_chunks` and `breakpoint_threshold_amount` were "
                "provided. `number_of_chunks` takes precedence."
            )

        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
            if breakpoint_threshold_amount is not None
            else _BREAKPOINT_DEFAULTS[breakpoint_threshold_type]
        )
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size

        # Backward-compatible aliases used by existing code/tests.
        self._embeddings = self.embeddings
        self._buffer_size = self.buffer_size
        self._add_start_index = add_start_index
        self._breakpoint_threshold_type = self.breakpoint_threshold_type
        self._breakpoint_threshold_amount = self.breakpoint_threshold_amount
        self._number_of_chunks = self.number_of_chunks
        self._sentence_split_regex = self.sentence_split_regex
        self._min_chunk_size = self.min_chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Split a single text into semantically coherent chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks ordered by their position in the original
            text.
        """
        chunk_ranges = self._split_text_to_ranges(text)
        return [text[r.start_index : r.end_index] for r in chunk_ranges]

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        """Split texts and wrap the resulting chunks as ``Document`` objects.

        Args:
            texts: A list of texts to split.
            metadatas: Optional per-text metadata dicts that are copied into
                every ``Document`` produced from the corresponding text.

        Returns:
            A flat list of ``Document`` objects across all input texts.
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents: list[Document] = []
        for i, text in enumerate(texts):
            chunk_ranges = self._split_text_to_ranges(text)
            for chunk_range in chunk_ranges:
                metadata = copy.deepcopy(metadatas_[i])
                if self._add_start_index:
                    metadata["start_index"] = chunk_range.start_index
                chunk_text = text[chunk_range.start_index : chunk_range.end_index]
                documents.append(Document(page_content=chunk_text, metadata=metadata))
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split ``Document`` objects and return new ``Document`` objects.

        Args:
            documents: The documents to split.

        Returns:
            A list of split ``Document`` objects.
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform documents by splitting them into semantic chunks.

        Args:
            documents: The sequence of documents to split.

        Returns:
            A sequence of split documents.
        """
        return self.split_documents(list(documents))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text_to_ranges(self, text: str) -> list[_ChunkRange]:
        """Split text into chunk ranges over the original text string.

        Args:
            text: The text to split.

        Returns:
            A list of ``_ChunkRange`` values that map each chunk back to
            character offsets in ``text``.
        """
        sentence_spans = self._split_into_sentences_with_spans(text)

        if not sentence_spans:
            return []

        if len(sentence_spans) == 1:
            sentence = sentence_spans[0]
            return [
                _ChunkRange(
                    start_index=sentence.start_index,
                    end_index=sentence.end_index,
                )
            ]

        # Preserve experimental behavior for 2-sentence gradient mode.
        if self.breakpoint_threshold_type == "gradient" and len(sentence_spans) == 2:  # noqa: PLR2004
            return [
                _ChunkRange(
                    start_index=sentence.start_index,
                    end_index=sentence.end_index,
                )
                for sentence in sentence_spans
            ]

        sentences = [sentence.sentence for sentence in sentence_spans]
        groups = _combine_sentences(sentences, buffer_size=self.buffer_size)
        combined_texts = [group.combined_sentence for group in groups]
        embeddings = self.embeddings.embed_documents(combined_texts)
        distances = _cosine_distances_between_consecutive(embeddings)

        breakpoint_indices = self._compute_breakpoint_indices(distances)
        chunk_ranges = self._assemble_chunk_ranges(sentence_spans, breakpoint_indices)
        if self.min_chunk_size is not None:
            chunk_ranges = self._merge_small_chunk_ranges(
                chunk_ranges,
                text=text,
                min_chunk_size=self.min_chunk_size,
            )
        return chunk_ranges

    def _split_into_sentences_with_spans(self, text: str) -> list[_SentenceSpan]:
        """Split text into sentence spans using the configured regex.

        Args:
            text: The raw text to split.

        Returns:
            Sentence spans with exact start/end offsets into the original text.
        """
        spans: list[_SentenceSpan] = []
        start_index = 0
        sentence_index = 0

        for match in re.finditer(self.sentence_split_regex, text):
            end_index = match.start()
            sentence = text[start_index:end_index]
            if sentence.strip():
                spans.append(
                    _SentenceSpan(
                        sentence=sentence,
                        sentence_index=sentence_index,
                        start_index=start_index,
                        end_index=end_index,
                    )
                )
                sentence_index += 1
            start_index = match.end()

        final_sentence = text[start_index:]
        if final_sentence.strip():
            spans.append(
                _SentenceSpan(
                    sentence=final_sentence,
                    sentence_index=sentence_index,
                    start_index=start_index,
                    end_index=len(text),
                )
            )

        return spans

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences using the configured regex.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of sentence strings. Empty/whitespace-only results are
            filtered out.
        """
        return [s.sentence for s in self._split_into_sentences_with_spans(text)]

    def _compute_breakpoint_indices(self, distances: list[float]) -> list[int]:
        """Determine which inter-sentence positions should become chunk boundaries.

        Args:
            distances: Cosine distances between consecutive sentence pairs.

        Returns:
            A sorted list of sentence indices *after* which a chunk boundary
            should be inserted.
        """
        if self.number_of_chunks is not None:
            threshold = self._threshold_for_target_chunks(distances)
        else:
            threshold = self._calculate_threshold(distances)

        return [i for i, d in enumerate(distances) if d > threshold]

    def _calculate_threshold(self, distances: list[float]) -> float:
        """Compute the breakpoint threshold using the configured strategy.

        Args:
            distances: Cosine distances between consecutive sentence pairs.

        Returns:
            The threshold value above which a distance is considered a
            chunk boundary.

        Raises:
            ValueError: If the configured threshold type is unknown.
        """
        amount = self.breakpoint_threshold_amount
        threshold_type = str(self.breakpoint_threshold_type)
        dist_array = np.array(distances)

        if threshold_type == "percentile":
            return float(np.percentile(dist_array, amount))

        if threshold_type == "standard_deviation":
            return float(np.mean(dist_array) + amount * np.std(dist_array))

        if threshold_type == "interquartile":
            q1 = float(np.percentile(dist_array, 25))
            q3 = float(np.percentile(dist_array, 75))
            iqr = q3 - q1
            return float(np.mean(dist_array) + amount * iqr)

        if threshold_type == "gradient":
            gradient = np.gradient(dist_array)
            return float(np.percentile(gradient, amount))

        msg = f"Got unexpected `breakpoint_threshold_type`: {threshold_type}"
        raise ValueError(msg)

    def _threshold_for_target_chunks(self, distances: list[float]) -> float:
        """Find the threshold that produces ``number_of_chunks`` chunks.

        Uses binary search over percentile values to converge on a threshold
        that yields the desired number of breakpoints (= chunks - 1).

        Args:
            distances: Cosine distances between consecutive sentence pairs.

        Returns:
            The threshold value that produces the closest match to the
            target number of chunks.
        """
        if self.number_of_chunks is None:
            msg = "This should never be called if `number_of_chunks` is None."
            raise ValueError(msg)

        target_breakpoints = self.number_of_chunks - 1
        dist_array = np.array(distances)

        # Edge case: more chunks requested than sentences allow.
        if target_breakpoints >= len(distances):
            return float(np.min(dist_array)) - 1e-10

        # Edge case: single chunk requested.
        if target_breakpoints <= 0:
            return float(np.max(dist_array)) + 1e-10

        # Binary search over percentile values [0, 100].
        low, high = 0.0, 100.0
        for _ in range(64):
            mid = (low + high) / 2.0
            threshold = float(np.percentile(dist_array, mid))
            num_breakpoints = int(np.sum(dist_array > threshold))
            if num_breakpoints < target_breakpoints:
                high = mid
            elif num_breakpoints > target_breakpoints:
                low = mid
            else:
                break

        return float(np.percentile(dist_array, mid))

    @staticmethod
    def _assemble_chunk_ranges(
        sentence_spans: list[_SentenceSpan], breakpoint_indices: list[int]
    ) -> list[_ChunkRange]:
        """Build chunk ranges according to breakpoint positions.

        Args:
            sentence_spans: Sentence spans from the source text.
            breakpoint_indices: Positions *after* which a chunk boundary
                should be placed.

        Returns:
            A list of chunk ranges mapping directly onto the original text.
        """
        chunk_ranges: list[_ChunkRange] = []
        start = 0
        for bp in sorted(breakpoint_indices):
            chunk_ranges.append(
                _ChunkRange(
                    start_index=sentence_spans[start].start_index,
                    end_index=sentence_spans[bp].end_index,
                )
            )
            start = bp + 1

        # Remaining sentences form the final chunk.
        if start < len(sentence_spans):
            chunk_ranges.append(
                _ChunkRange(
                    start_index=sentence_spans[start].start_index,
                    end_index=sentence_spans[-1].end_index,
                )
            )

        return chunk_ranges

    @staticmethod
    def _merge_small_chunk_ranges(
        chunk_ranges: list[_ChunkRange],
        *,
        text: str,
        min_chunk_size: int,
    ) -> list[_ChunkRange]:
        """Merge undersized chunk ranges into neighboring chunk ranges."""
        if not chunk_ranges:
            return []

        merged: list[_ChunkRange] = [chunk_ranges[0]]
        for chunk_range in chunk_ranges[1:]:
            previous = merged[-1]
            previous_text = text[previous.start_index : previous.end_index]
            if len(previous_text) < min_chunk_size:
                merged[-1] = _ChunkRange(
                    start_index=previous.start_index,
                    end_index=chunk_range.end_index,
                )
            else:
                merged.append(chunk_range)

        if len(merged) > 1:
            last = merged[-1]
            last_text = text[last.start_index : last.end_index]
            if len(last_text) < min_chunk_size:
                last = merged.pop()
                previous = merged[-1]
                merged[-1] = _ChunkRange(
                    start_index=previous.start_index,
                    end_index=last.end_index,
                )

        return merged
