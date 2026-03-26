"""Unit tests for the SemanticSimilarityTextSplitter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_text_splitters.semantic import (
    SemanticSimilarityTextSplitter,
    _cosine_similarity,
    _split_sentences,
)


def _make_fake_embeddings(
    similarities: list[float],
) -> Any:
    """Build a mock embedding model.

    Produces vectors that result in the requested pairwise cosine similarities
    between adjacent vectors.

    Args:
        similarities: Target similarities between neighbors.

    Returns:
        Mock object with embed_documents method.
    """
    import math

    n = len(similarities) + 1
    vectors: list[list[float]] = []
    for i in range(n):
        angle = sum(math.acos(max(-1.0, min(1.0, s))) for s in similarities[:i])
        vec = [math.cos(angle), math.sin(angle)]
        vectors.append(vec)

    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: vectors[: len(texts)]
    return mock


class _FixedEmbeddings:
    """Mock Embeddings that returns caller-supplied vectors."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return list(self._vectors[: len(texts)])

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[0]


class TestCosineSimilarity:
    """Test the internal cosine similarity calculation."""
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_general_case(self) -> None:
        result = _cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert result == pytest.approx(0.7071, abs=1e-3)


class TestSplitSentences:
    """Test the built-in sentence splitting heuristic."""
    def test_basic_sentences(self) -> None:
        text = "Hello world. How are you? I am fine!"
        result = _split_sentences(text)
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_single_sentence(self) -> None:
        text = "Just one sentence"
        result = _split_sentences(text)
        assert result == ["Just one sentence"]

    def test_empty_string(self) -> None:
        result = _split_sentences("")
        assert result == []

    def test_strips_whitespace(self) -> None:
        text = "  Hello.  World.  "
        result = _split_sentences(text)
        assert all(s == s.strip() for s in result)


class TestSemanticSplitterInit:
    """Test Initialization and parameter validation."""
    def test_default_construction(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(embeddings=mock_emb)
        assert splitter._breakpoint_threshold_type == "percentile"
        assert splitter._breakpoint_threshold_amount == 95.0

    def test_standard_deviation_default_amount(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(
            embeddings=mock_emb,
            breakpoint_threshold_type="standard_deviation",
        )
        assert splitter._breakpoint_threshold_amount == 3.0

    def test_interquartile_default_amount(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(
            embeddings=mock_emb,
            breakpoint_threshold_type="interquartile",
        )
        assert splitter._breakpoint_threshold_amount == 1.5

    def test_custom_threshold_amount(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(
            embeddings=mock_emb,
            breakpoint_threshold_amount=80.0,
        )
        assert splitter._breakpoint_threshold_amount == 80.0

    def test_invalid_threshold_type_raises(self) -> None:
        mock_emb = MagicMock()
        with pytest.raises(ValueError, match="Invalid breakpoint_threshold_type"):
            SemanticSimilarityTextSplitter(
                embeddings=mock_emb,
                breakpoint_threshold_type="unknown",  # type: ignore[arg-type]
            )

    def test_negative_threshold_amount_raises(self) -> None:
        mock_emb = MagicMock()
        with pytest.raises(ValueError, match="breakpoint_threshold_amount must be"):
            SemanticSimilarityTextSplitter(
                embeddings=mock_emb,
                breakpoint_threshold_amount=-1.0,
            )


class TestSemanticSplitterSplitText:
    """Test core splitting logic with various thresholds."""
    def test_empty_text_returns_empty(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(embeddings=mock_emb)
        assert splitter.split_text("") == []

    def test_single_sentence_returned_as_one_chunk(self) -> None:
        mock_emb = MagicMock()
        splitter = SemanticSimilarityTextSplitter(embeddings=mock_emb)
        result = splitter.split_text("Just one sentence here")
        assert len(result) == 1
        assert "Just one sentence here" in result[0]

    def test_high_similarity_keeps_sentences_together(self) -> None:
        sentences = [
            "The cat sat on the mat.",
            "The cat rested on the mat.",
            "The cat lay on the mat.",
        ]
        vectors = [[1.0, 0.0]] * len(sentences)
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,
        )
        result = splitter.split_text(" ".join(sentences))
        assert len(result) == 1

    def test_topic_shift_produces_two_chunks(self) -> None:
        import math

        group_a = [1.0, 0.0]
        group_b = [0.0, 1.0]
        vectors = [group_a, group_a, group_b, group_b]

        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50.0,
        )
        text = (
            "Dogs are loyal animals. "
            "They are known for their friendship. "
            "Stars are massive celestial bodies. "
            "They emit light through nuclear fusion."
        )
        result = splitter.split_text(text)
        assert len(result) == 2

    def test_custom_sentence_split_regex(self) -> None:
        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            sentence_split_regex=r";",
        )
        text = "First part; Second part; Third part"
        result = splitter.split_text(text)
        assert isinstance(result, list)

    def test_split_documents_integration(self) -> None:
        from langchain_core.documents import Document

        vectors = [[1.0, 0.0], [0.0, 1.0]]
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50.0,
        )
        doc = Document(page_content="First sentence. Second sentence.")
        docs = splitter.split_documents([doc])
        assert all(isinstance(d, Document) for d in docs)

    @pytest.mark.parametrize(
        "threshold_type",
        ["percentile", "standard_deviation", "interquartile"],
    )
    def test_all_threshold_types_do_not_crash(self, threshold_type: str) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type=threshold_type,
        )
        result = splitter.split_text("Sentence one. Sentence two. Sentence three.")
        assert isinstance(result, list)


class TestHierarchicalSemanticSplitting:
    """Test recursive size-based splitting fallback."""
    def test_topic_break_respected_first(self) -> None:

        vectors = [[1.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb, chunk_size=1000, breakpoint_threshold_amount=50
        )
        text = "S1. S2. S3. S4. S5. S6."
        result = splitter.split_text(text)
        assert len(result) == 3

    def test_recursive_split_on_size_violation(self) -> None:
        vectors = [[1.0, 0.0]] * 6
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb, chunk_size=8, chunk_overlap=0, breakpoint_threshold_amount=95
        )
        text = "S1. S2. S3. S4. S5. S6."
        result = splitter.split_text(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= 8

    def test_does_not_split_single_sentence_even_if_oversized(self) -> None:
        emb = _FixedEmbeddings([[1.0, 0.0]])
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb, chunk_size=10, chunk_overlap=0
        )
        text = "This is a very long sentence."
        result = splitter.split_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_handles_redundant_similarities(self) -> None:
        emb = _FixedEmbeddings([[1.0, 0.0]] * 4)
        splitter = SemanticSimilarityTextSplitter(embeddings=emb, chunk_size=1000)
        result = splitter.split_text("S1. S2. S3. S4.")
        assert len(result) == 1
