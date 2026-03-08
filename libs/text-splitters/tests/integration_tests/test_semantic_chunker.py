"""Integration tests for SemanticChunker.

These tests require ``numpy`` and use ``DeterministicFakeEmbedding`` from
``langchain-core`` to run without external API keys while still exercising the
full pipeline end-to-end with real numpy operations.
"""

from __future__ import annotations

import random

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from langchain_text_splitters.semantic import SemanticChunker


@pytest.mark.requires("numpy")
class TestSemanticChunkerIntegration:
    """End-to-end tests with deterministic (but realistic) embeddings."""

    def test_basic_split(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=128)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It allows systems to learn from data. "
            "Deep learning uses neural networks with many layers. "
            "The weather today is sunny with clear skies. "
            "Tomorrow it might rain in the afternoon. "
            "Pack an umbrella just in case."
        )
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1
        # All content preserved.
        joined = " ".join(chunks)
        assert "Machine learning" in joined
        assert "umbrella" in joined

    def test_long_document(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=64)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="standard_deviation",
        )
        # Generate a longer document with distinct topics.
        paragraphs = [
            "Python is a programming language. It supports multiple paradigms. "
            "Python is widely used in data science.",
            "The ocean is vast and deep. Marine life is incredibly diverse. "
            "Coral reefs are under threat from climate change.",
            "Music has been part of human culture for millennia. "
            "Instruments range from simple drums to complex synthesizers. "
            "Genres evolve and blend over time.",
        ]
        text = " ".join(paragraphs)
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_create_documents_round_trip(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=64)
        chunker = SemanticChunker(embeddings, add_start_index=True)
        text = "Alpha sentence. Beta sentence. Gamma sentence."
        docs = chunker.create_documents([text], metadatas=[{"source": "test"}])
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata["source"] == "test" for d in docs)
        for doc in docs:
            idx = doc.metadata["start_index"]
            assert text[idx:].startswith(doc.page_content)

    def test_number_of_chunks_mode(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=64)
        chunker = SemanticChunker(embeddings, number_of_chunks=2)
        text = (
            "Stars are massive balls of gas. "
            "They produce energy through nuclear fusion. "
            "Our sun is a medium-sized star. "
            "Computers process information using binary logic. "
            "Modern CPUs contain billions of transistors. "
            "Software controls how hardware operates."
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 2


class _IndexedRandomEmbedding(Embeddings):
    """Deterministic embeddings keyed by sentence index for stress testing."""

    def __init__(self, *, size: int = 64) -> None:
        self._size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(len(texts)):
            rng = random.Random(i)
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(self._size)])
        return vectors

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * self._size


def _assert_exact_offsets(text: str, docs: list[Document]) -> None:
    previous_end = -1
    for doc in docs:
        start_index = doc.metadata["start_index"]
        assert isinstance(start_index, int)
        assert 0 <= start_index < len(text)
        assert start_index >= previous_end
        end_index = start_index + len(doc.page_content)
        assert text[start_index:end_index] == doc.page_content
        previous_end = end_index


def _build_large_html_document(*, sections: int = 240) -> str:
    parts: list[str] = []
    for i in range(sections):
        section_id = f"section-{i:03d}"
        parts.extend(
            [
                f"<section id='{section_id}'>\n",
                f"<h2>{section_id}</h2>\n",
                (
                    f"<p>{section_id} summarizes retrieval quality metrics. "
                    "Analysts compared answer grounding across corpora. "
                    "The baseline model missed cross-paragraph evidence.</p>\n"
                ),
                (
                    f"<p>The remediation plan for {section_id} emphasizes semantic "
                    "boundaries. This change reduced context fragmentation in long "
                    "pages. Follow-up audits confirmed stable precision.</p>\n"
                ),
                "</section>\n\n",
            ]
        )
    return "".join(parts)


def _build_chat_transcript(*, turns: int = 700) -> str:
    lines: list[str] = []
    for i in range(turns):
        speaker = "user" if i % 2 == 0 else "assistant"
        lines.append(
            f"turn-{i:04d} [{speaker}]: I observed incident bucket {i % 23}. "
            f"We should update remediation checklist shard {i % 31} before "
            "deployment.\n"
        )
    return "".join(lines)


@pytest.mark.requires("numpy")
class TestSemanticChunkerStress:
    def test_stress_large_html_document_offsets_and_markers(self) -> None:
        text = _build_large_html_document(sections=240)
        chunker = SemanticChunker(
            DeterministicFakeEmbedding(size=128),
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,
            min_chunk_size=600,
        )
        docs = chunker.create_documents([text], metadatas=[{"source": "html-stress"}])
        assert len(docs) > 50
        assert all(doc.metadata["source"] == "html-stress" for doc in docs)
        _assert_exact_offsets(text, docs)

        joined = "".join(doc.page_content for doc in docs)
        for marker in ("section-000", "section-120", "section-239"):
            assert marker in joined

    def test_stress_long_chat_transcript_with_target_chunks(self) -> None:
        text = _build_chat_transcript(turns=700)
        chunker_a = SemanticChunker(
            _IndexedRandomEmbedding(size=96),
            add_start_index=True,
            number_of_chunks=40,
            breakpoint_threshold_amount=99,
            min_chunk_size=300,
        )
        chunker_b = SemanticChunker(
            _IndexedRandomEmbedding(size=96),
            add_start_index=True,
            number_of_chunks=40,
            breakpoint_threshold_amount=5,
            min_chunk_size=300,
        )

        docs_a = chunker_a.create_documents([text], metadatas=[{"source": "chat"}])
        docs_b = chunker_b.create_documents([text], metadatas=[{"source": "chat"}])
        assert len(docs_a) == len(docs_b)
        assert 10 <= len(docs_a) <= 40
        _assert_exact_offsets(text, docs_a)
        _assert_exact_offsets(text, docs_b)

        joined = "".join(doc.page_content for doc in docs_a)
        for marker in ("turn-0000", "turn-0350", "turn-0699"):
            assert marker in joined

    def test_stress_batch_many_documents_metadata_and_offsets(self) -> None:
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        for doc_id in range(180):
            segments = [
                (
                    f"doc-{doc_id:03d} segment-{segment:02d} discusses retrieval "
                    "precision changes after semantic splitting."
                )
                for segment in range(12)
            ]
            texts.append(".\n".join(segments) + ".")
            metadatas.append({"doc_id": doc_id, "nested": {"source": "stress"}})

        chunker = SemanticChunker(
            DeterministicFakeEmbedding(size=72),
            add_start_index=True,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=1.2,
            min_chunk_size=180,
        )
        docs = chunker.create_documents(texts, metadatas=metadatas)
        assert len(docs) >= len(texts)

        source_text_by_id = dict(enumerate(texts))
        for doc in docs:
            doc_id = doc.metadata["doc_id"]
            source_text = source_text_by_id[doc_id]
            start_index = doc.metadata["start_index"]
            end_index = start_index + len(doc.page_content)
            assert source_text[start_index:end_index] == doc.page_content
            assert doc.metadata["nested"]["source"] == "stress"

        sample_doc_id = docs[0].metadata["doc_id"]
        sample_docs = [doc for doc in docs if doc.metadata["doc_id"] == sample_doc_id]
        assert len(sample_docs) >= 1
        sample_docs[0].metadata["nested"]["mutated"] = True
        for other in sample_docs[1:]:
            assert "mutated" not in other.metadata["nested"]

    def test_realistic_html_like_text_preserves_exact_offsets(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=96)
        chunker = SemanticChunker(
            embeddings,
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=60,
        )
        text = (
            "<h1>Clinical Trial Summary</h1>\n"
            "<p>Participants received treatment A for six weeks.</p>\n\n"
            "<p>Primary endpoint was reduction in symptom severity by week eight.</p>\n"
            "<p>Adverse events were mild and self-limited in most cases.</p>\n\n"
            "<h2>Methods</h2>\n"
            "<p>Randomization was stratified by age and baseline biomarkers.</p>\n"
            "<p>Missing values were imputed using multiple imputation.</p>\n"
        )
        docs = chunker.create_documents([text], metadatas=[{"source": "html"}])
        assert len(docs) >= 1
        for doc in docs:
            idx = doc.metadata["start_index"]
            assert idx >= 0
            assert doc.metadata["source"] == "html"
            assert text[idx : idx + len(doc.page_content)] == doc.page_content

    def test_realistic_scientific_text_min_chunk_size_preserves_content(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=128)
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=20,
            min_chunk_size=180,
        )
        text = (
            "Background. Retrieval-augmented generation systems depend on chunk "
            "quality. "
            "Simple fixed-size chunking can separate related findings. "
            "Methods. We compare recursive and semantic chunking over biomedical "
            "abstracts. "
            "The semantic method computes sentence embeddings and identifies topic "
            "shifts. "
            "Results. Semantic chunks improved retrieval precision and reduced "
            "hallucination rates. "
            "Conclusion. Structure-aware segmentation helps preserve context integrity."
        )
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1
        joined = "".join(chunks)
        assert (
            "Retrieval-augmented generation systems depend on chunk quality." in joined
        )
        assert (
            "Structure-aware segmentation helps preserve context integrity." in joined
        )
        if len(chunks) > 1:
            assert all(len(chunk) >= 180 for chunk in chunks[:-1])

    def test_number_of_chunks_takes_precedence_over_breakpoint_amount(self) -> None:
        embeddings = DeterministicFakeEmbedding(size=64)
        chunker = SemanticChunker(
            embeddings,
            number_of_chunks=2,
            breakpoint_threshold_amount=99,
        )
        text = (
            "Sentence one. Sentence two. Sentence three. Sentence four. "
            "Sentence five. Sentence six."
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 2
