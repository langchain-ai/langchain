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


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_fake_embeddings(
    similarities: list[float],
) -> Any:
    """Build a mock `Embeddings` whose `embed_documents` returns vectors that
    produce the requested pairwise cosine similarities.

    Strategy: use N-dimensional orthonormal basis vectors and blend them so
    that adjacent pairs have the desired similarity.  For simplicity here we
    just return pre-built numeric vectors that the real cosine function will
    score correctly.

    For unit tests, we manufacture vectors directly by returning
    `[[1, 0, 0, ...], [cos, sin, 0, ...], ...]` where each pair has the
    given similarity by construction.
    """
    import math

    # Build vectors so that dot(v[i], v[i+1]) == similarities[i].
    # Use 2-D plane rotation trick: v[i+1] = cos_angle * x_hat + sin_angle * y_hat
    # We need N+1 vectors for N similarities.
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


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_general_case(self) -> None:
        # [1, 1] and [1, 0] → cos(45°) ≈ 0.7071
        result = _cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert result == pytest.approx(0.7071, abs=1e-3)


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
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


# ---------------------------------------------------------------------------
# SemanticSimilarityTextSplitter – construction
# ---------------------------------------------------------------------------


class TestSemanticSplitterInit:
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


# ---------------------------------------------------------------------------
# SemanticSimilarityTextSplitter – split_text
# ---------------------------------------------------------------------------


class TestSemanticSplitterSplitText:
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
        # All consecutive sentences are very similar (sim = 1.0), so no splits.
        sentences = [
            "The cat sat on the mat.",
            "The cat rested on the mat.",
            "The cat lay on the mat.",
        ]
        # Use identical vectors → similarity = 1.0 everywhere
        vectors = [[1.0, 0.0]] * len(sentences)
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,
        )
        result = splitter.split_text(" ".join(sentences))
        # All sentences similar → should produce 1 chunk
        assert len(result) == 1

    def test_topic_shift_produces_two_chunks(self) -> None:
        # Two clearly distinct topic blocks: first two sentences are about
        # dogs, last two are about astronomy.  We fake low similarity between
        # sentence 2 and 3, high within each group.
        import math

        # Vectors: 0° for group A, 90° for group B → sim(A,B) = cos(90°) = 0
        group_a = [1.0, 0.0]
        group_b = [0.0, 1.0]
        vectors = [group_a, group_a, group_b, group_b]

        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50.0,  # aggressive split
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
        # Split on semicolons instead of punctuation.
        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        emb = _FixedEmbeddings(vectors)
        splitter = SemanticSimilarityTextSplitter(
            embeddings=emb,
            sentence_split_regex=r";",
        )
        text = "First part; Second part; Third part"
        # All high-similarity → 1 chunk, but it should not crash.
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
            breakpoint_threshold_type=threshold_type,  # type: ignore[arg-type]
        )
        result = splitter.split_text("Sentence one. Sentence two. Sentence three.")
        assert isinstance(result, list)
