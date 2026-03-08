"""Unit tests for SemanticChunker."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_text_splitters.semantic import (
    SemanticChunker,
    _combine_sentences,
    _cosine_distances_between_consecutive,
    _merge_small_chunks,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _IdentityEmbeddings(Embeddings):
    """Embeddings that assign a unique unit vector per sentence position.

    Sentence *i* is mapped to a one-hot-like vector with a 1.0 at dimension *i*.
    This gives maximum cosine distance (1.0) between all non-identical sentences
    and 0.0 distance for identical ones — useful for controlled breakpoint tests.
    """

    def __init__(self, *, dims: int = 64) -> None:
        self._dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(len(texts)):
            vec = [0.0] * self._dims
            vec[i % self._dims] = 1.0
            result.append(vec)
        return result

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * self._dims


class _ClusterEmbeddings(Embeddings):
    """Embeddings that cluster sentences into groups via a mapping.

    Sentences in the same cluster share the same embedding vector, so cosine
    distance within a cluster is 0 and between clusters is 1. This lets us
    deterministically control where breakpoints should appear.

    ``cluster_map`` maps sentence index -> cluster id. Cluster ids are used
    as the non-zero dimension index in the embedding vector.
    """

    def __init__(self, cluster_map: list[int], *, dims: int = 64) -> None:
        self._cluster_map = cluster_map
        self._dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(len(texts)):
            vec = [0.0] * self._dims
            cluster = self._cluster_map[i] if i < len(self._cluster_map) else 0
            vec[cluster % self._dims] = 1.0
            result.append(vec)
        return result

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * self._dims


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestCombineSentences:
    def test_buffer_zero(self) -> None:
        groups = _combine_sentences(["A", "B", "C"], buffer_size=0)
        assert [g.combined_sentence for g in groups] == ["A", "B", "C"]

    def test_buffer_one(self) -> None:
        groups = _combine_sentences(["A", "B", "C"], buffer_size=1)
        assert groups[0].combined_sentence == "A B"
        assert groups[1].combined_sentence == "A B C"
        assert groups[2].combined_sentence == "B C"

    def test_buffer_larger_than_list(self) -> None:
        groups = _combine_sentences(["A", "B"], buffer_size=5)
        assert groups[0].combined_sentence == "A B"
        assert groups[1].combined_sentence == "A B"

    def test_preserves_original_sentence(self) -> None:
        groups = _combine_sentences(["Hello world", "Goodbye"], buffer_size=1)
        assert groups[0].sentence == "Hello world"
        assert groups[1].sentence == "Goodbye"

    def test_index_tracking(self) -> None:
        groups = _combine_sentences(["A", "B", "C"], buffer_size=0)
        assert [g.sentence_index for g in groups] == [0, 1, 2]

    def test_empty_list(self) -> None:
        assert _combine_sentences([], buffer_size=1) == []

    def test_single_sentence(self) -> None:
        groups = _combine_sentences(["Only one"], buffer_size=1)
        assert len(groups) == 1
        assert groups[0].combined_sentence == "Only one"


@pytest.mark.requires("numpy")
class TestCosineDistances:
    def test_identical_embeddings(self) -> None:
        embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        distances = _cosine_distances_between_consecutive(embeddings)
        assert len(distances) == 2
        assert all(math.isclose(d, 0.0, abs_tol=1e-9) for d in distances)

    def test_orthogonal_embeddings(self) -> None:
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        distances = _cosine_distances_between_consecutive(embeddings)
        assert len(distances) == 2
        assert math.isclose(distances[0], 1.0, abs_tol=1e-9)
        assert math.isclose(distances[1], 1.0, abs_tol=1e-9)

    def test_opposite_embeddings(self) -> None:
        embeddings = [[1.0, 0.0], [-1.0, 0.0]]
        distances = _cosine_distances_between_consecutive(embeddings)
        assert math.isclose(distances[0], 2.0, abs_tol=1e-9)

    def test_zero_vector_handled(self) -> None:
        embeddings = [[0.0, 0.0], [1.0, 0.0]]
        distances = _cosine_distances_between_consecutive(embeddings)
        assert len(distances) == 1
        # Zero vector after normalization gives 0 similarity.
        assert 0.0 <= distances[0] <= 2.0


@pytest.mark.requires("numpy")
class TestMergeSmallChunks:
    def test_no_merge_needed(self) -> None:
        chunks = ["Hello world", "Goodbye world"]
        result = _merge_small_chunks(chunks, min_chunk_size=5)
        assert result == chunks

    def test_merge_into_previous(self) -> None:
        chunks = ["Hello world", "Hi", "Goodbye world"]
        result = _merge_small_chunks(chunks, min_chunk_size=5)
        assert result == ["Hello world", "Hi Goodbye world"]

    def test_merge_first_chunk_forward(self) -> None:
        chunks = ["Hi", "Hello world", "Goodbye world"]
        result = _merge_small_chunks(chunks, min_chunk_size=5)
        assert result == ["Hi Hello world", "Goodbye world"]

    def test_merge_last_chunk_backward(self) -> None:
        chunks = ["Hello world", "Goodbye world", "Hi"]
        result = _merge_small_chunks(chunks, min_chunk_size=5)
        assert result == ["Hello world", "Goodbye world Hi"]

    def test_empty_input(self) -> None:
        assert _merge_small_chunks([], min_chunk_size=5) == []

    def test_single_small_chunk(self) -> None:
        result = _merge_small_chunks(["Hi"], min_chunk_size=100)
        assert result == ["Hi"]

    def test_all_small_chunks_collapse(self) -> None:
        chunks = ["A", "B", "C"]
        result = _merge_small_chunks(chunks, min_chunk_size=10)
        assert len(result) == 1
        assert result == ["A B C"]


# ---------------------------------------------------------------------------
# Tests for SemanticChunker
# ---------------------------------------------------------------------------


@pytest.mark.requires("numpy")
class TestSemanticChunkerInit:
    def test_default_init(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        assert chunker.breakpoint_threshold_type == "percentile"
        assert chunker.breakpoint_threshold_amount == 95
        assert chunker.buffer_size == 1
        assert chunker.number_of_chunks is None
        assert chunker.min_chunk_size is None
        # Backward-compatible private aliases.
        assert chunker._breakpoint_threshold_type == "percentile"
        assert chunker._breakpoint_threshold_amount == 95

    def test_mutual_exclusion_preserves_experimental_precedence(self) -> None:
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            number_of_chunks=3,
            breakpoint_threshold_amount=90,
        )
        assert chunker.number_of_chunks == 3
        assert chunker.breakpoint_threshold_amount == 90

    def test_positional_args_supported_for_backward_compatibility(self) -> None:
        add_start_index = True
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            2,
            add_start_index,
            "percentile",
            90,
            None,
            r"(?<=[.?!])\s+",
            20,
        )
        assert chunker.buffer_size == 2
        assert chunker._add_start_index is True
        assert chunker.min_chunk_size == 20

    def test_invalid_breakpoint_threshold_type(self) -> None:
        with pytest.raises(ValueError, match="unexpected `breakpoint_threshold_type`"):
            SemanticChunker(
                _IdentityEmbeddings(),
                breakpoint_threshold_type="bad",  # type: ignore[arg-type]
            )

    def test_invalid_buffer_size(self) -> None:
        with pytest.raises(ValueError, match="buffer_size"):
            SemanticChunker(_IdentityEmbeddings(), buffer_size=-1)

    def test_invalid_number_of_chunks(self) -> None:
        with pytest.raises(ValueError, match="number_of_chunks"):
            SemanticChunker(_IdentityEmbeddings(), number_of_chunks=0)

    def test_invalid_min_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="min_chunk_size"):
            SemanticChunker(_IdentityEmbeddings(), min_chunk_size=0)

    def test_numpy_not_installed(self) -> None:
        with (
            patch("langchain_text_splitters.semantic._HAS_NUMPY", new=False),
            pytest.raises(ImportError, match="numpy"),
        ):
            SemanticChunker(_IdentityEmbeddings())


@pytest.mark.requires("numpy")
class TestSemanticChunkerSplitText:
    def test_empty_text(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        assert chunker.split_text("") == []

    def test_whitespace_only(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        assert chunker.split_text("   \n\t  ") == []

    def test_single_sentence(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        result = chunker.split_text("Hello world.")
        assert result == ["Hello world."]

    def test_two_sentences_returns_single_chunk(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        text = "First sentence. Second sentence."
        result = chunker.split_text(text)
        assert result == [text]

    def test_two_sentences_gradient_matches_experimental_behavior(self) -> None:
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            breakpoint_threshold_type="gradient",
        )
        text = "First sentence. Second sentence."
        result = chunker.split_text(text)
        assert result == ["First sentence.", "Second sentence."]

    def test_clear_topic_boundary(self) -> None:
        """Verify that a clear topic change produces a chunk boundary.

        Sentences in the same cluster embed identically (distance=0), while
        sentences in different clusters are orthogonal (distance=1). With
        percentile=50, the boundary between clusters should be detected.
        """
        # Cluster 0: sentences 0,1,2  |  Cluster 1: sentences 3,4,5
        cluster_map = [0, 0, 0, 1, 1, 1]
        embeddings = _ClusterEmbeddings(cluster_map)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        text = "A cat sat. A cat played. A cat slept. Dogs run. Dogs bark. Dogs play."
        result = chunker.split_text(text)
        assert len(result) == 2
        assert "cat" in result[0]
        assert "Dogs" in result[1]

    def test_all_identical_produces_single_chunk(self) -> None:
        """When all embeddings are identical, no breakpoints are found."""

        class _ConstantEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0]] * len(texts)

            def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
                return [1.0, 0.0, 0.0]

        chunker = SemanticChunker(
            _ConstantEmbeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        text = "First point. Second point. Third point. Fourth point."
        result = chunker.split_text(text)
        assert len(result) == 1

    def test_custom_regex(self) -> None:
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            sentence_split_regex=r"\n",
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0,
        )
        text = "Line one\nLine two\nLine three"
        result = chunker.split_text(text)
        assert len(result) >= 1
        # All content preserved.
        full = " ".join(result)
        assert "Line one" in full
        assert "Line two" in full
        assert "Line three" in full


@pytest.mark.requires("numpy")
class TestSemanticChunkerThresholdTypes:
    """Verify each threshold type runs without error and produces chunks."""

    _text = "The sky is blue. Water is wet. Fire is hot. Ice is cold. Rocks are hard."

    @pytest.mark.parametrize(
        "threshold_type",
        ["percentile", "standard_deviation", "interquartile", "gradient"],
    )
    def test_threshold_type_runs(self, threshold_type: str) -> None:
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            breakpoint_threshold_type=threshold_type,  # type: ignore[arg-type]
        )
        result = chunker.split_text(self._text)
        assert len(result) >= 1
        # All original content should be preserved across chunks.
        joined = " ".join(result)
        for sentence_fragment in ["sky", "Water", "Fire", "Ice", "Rocks"]:
            assert sentence_fragment in joined

    def test_invalid_threshold_type_raises_even_with_amount(self) -> None:
        with pytest.raises(ValueError, match="unexpected `breakpoint_threshold_type`"):
            SemanticChunker(
                _IdentityEmbeddings(),
                breakpoint_threshold_type="bad",  # type: ignore[arg-type]
                breakpoint_threshold_amount=90,
            )


@pytest.mark.requires("numpy")
class TestSemanticChunkerNumberOfChunks:
    def test_exact_number(self) -> None:
        cluster_map = [0, 0, 1, 1, 2, 2]
        embeddings = _ClusterEmbeddings(cluster_map)
        chunker = SemanticChunker(embeddings, number_of_chunks=3)
        text = "A one. A two. B one. B two. C one. C two."
        result = chunker.split_text(text)
        assert len(result) == 3

    def test_more_chunks_than_sentences(self) -> None:
        """Requesting more chunks than sentences returns each sentence."""
        cluster_map = [0, 1, 2]
        embeddings = _ClusterEmbeddings(cluster_map)
        chunker = SemanticChunker(embeddings, number_of_chunks=10)
        text = "First. Second. Third."
        result = chunker.split_text(text)
        assert len(result) == 3

    def test_single_chunk(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings(), number_of_chunks=1)
        text = "One. Two. Three. Four."
        result = chunker.split_text(text)
        assert len(result) == 1


@pytest.mark.requires("numpy")
class TestSemanticChunkerMinChunkSize:
    def test_small_chunks_merged(self) -> None:
        cluster_map = [0, 1, 2, 3, 4]
        embeddings = _ClusterEmbeddings(cluster_map)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0,
            min_chunk_size=50,
        )
        text = "A. B. C. D. E."
        result = chunker.split_text(text)
        # All chunks should meet min_chunk_size or be the sole chunk.
        for chunk in result:
            if len(result) > 1:
                assert len(chunk) >= 50 or chunk == result[-1]

    def test_content_preserved(self) -> None:
        cluster_map = [0, 1, 2]
        embeddings = _ClusterEmbeddings(cluster_map)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0,
            min_chunk_size=100,
        )
        text = "Short. Also short. Yet another."
        result = chunker.split_text(text)
        joined = " ".join(result)
        assert "Short" in joined
        assert "Also short" in joined
        assert "Yet another" in joined


@pytest.mark.requires("numpy")
class TestSemanticChunkerDocuments:
    def test_create_documents(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        text = "Hello world."
        docs = chunker.create_documents([text])
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].page_content == "Hello world."

    def test_create_documents_with_metadata(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        docs = chunker.create_documents(
            ["One sentence."], metadatas=[{"source": "test"}]
        )
        assert docs[0].metadata["source"] == "test"

    def test_metadata_isolation(self) -> None:
        """Metadata dicts should be deep-copied so mutations don't propagate."""
        chunker = SemanticChunker(
            _IdentityEmbeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0,
        )
        meta = {"source": "test"}
        text = "First. Second. Third."
        docs = chunker.create_documents([text], metadatas=[meta])
        if len(docs) > 1:
            docs[0].metadata["extra"] = "mutated"
            assert "extra" not in docs[1].metadata

    def test_add_start_index(self) -> None:
        chunker = SemanticChunker(
            _ClusterEmbeddings([0, 0, 1, 1]),
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        text = "Cat sat. Cat played. Dog ran. Dog barked."
        docs = chunker.create_documents([text])
        for doc in docs:
            assert "start_index" in doc.metadata
            idx = doc.metadata["start_index"]
            assert text[idx:].startswith(doc.page_content)

    def test_add_start_index_with_newlines_preserves_exact_offsets(self) -> None:
        chunker = SemanticChunker(
            _ClusterEmbeddings([0, 0, 1]),
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0,
        )
        text = "Alpha sentence.\n\nBeta sentence.\nGamma sentence."
        docs = chunker.create_documents([text])
        assert len(docs) >= 1
        for doc in docs:
            idx = doc.metadata["start_index"]
            assert idx >= 0
            assert text[idx : idx + len(doc.page_content)] == doc.page_content

    def test_split_documents(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        input_docs = [
            Document(page_content="Hello world.", metadata={"source": "a"}),
        ]
        result = chunker.split_documents(input_docs)
        assert len(result) >= 1
        assert result[0].metadata["source"] == "a"

    def test_transform_documents(self) -> None:
        chunker = SemanticChunker(_IdentityEmbeddings())
        input_docs = [Document(page_content="Hello world.")]
        result = chunker.transform_documents(input_docs)
        assert len(result) >= 1


@pytest.mark.requires("numpy")
class TestSemanticChunkerBufferSize:
    def test_buffer_zero_vs_one(self) -> None:
        """Different buffer sizes should not lose content."""
        text = "A. B. C. D. E."
        for bs in (0, 1, 2):
            chunker = SemanticChunker(_IdentityEmbeddings(), buffer_size=bs)
            result = chunker.split_text(text)
            joined = " ".join(result)
            for fragment in ["A", "B", "C", "D", "E"]:
                assert fragment in joined
