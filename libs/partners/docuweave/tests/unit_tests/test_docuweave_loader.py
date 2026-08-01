"""Unit tests for DocuWeaveLoader."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

docuweave = pytest.importorskip("docuweave")

from langchain_docuweave import DocuWeaveLoader  # noqa: E402


def _make_chunk(i: int, section: str = "Introduction") -> dict:
    return {
        "id": f"c_{i:04d}",
        "text": f"Chunk {i} body text.",
        "tokens": 4,
        "section_title": section,
        "section_path": section,
        "section_level": 0,
        "page_start": i + 1,
        "page_end": i + 1,
        "previous_chunk_id": f"c_{i - 1:04d}" if i > 0 else None,
        "next_chunk_id": f"c_{i + 1:04d}",
    }


def _mock_doc(num_chunks: int = 3, confidence: float = 0.75) -> MagicMock:
    doc = MagicMock()
    doc.hierarchy_confidence = confidence
    doc.iter_chunks.return_value = iter([_make_chunk(i) for i in range(num_chunks)])
    return doc


class TestDocuWeaveLoaderInit:
    def test_stores_file_path_as_string(self) -> None:
        loader = DocuWeaveLoader("paper.pdf")
        assert loader.file_path == "paper.pdf"

    def test_path_object_converted_to_str(self) -> None:
        loader = DocuWeaveLoader(Path("/tmp/paper.pdf"))
        assert loader.file_path == "/tmp/paper.pdf"

    def test_default_max_tokens(self) -> None:
        assert DocuWeaveLoader("paper.pdf").max_tokens == 512

    def test_custom_max_tokens(self) -> None:
        assert DocuWeaveLoader("paper.pdf", max_tokens=256).max_tokens == 256


class TestDocuWeaveLoaderLazyLoad:
    def test_yields_correct_number_of_documents(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(3)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert len(docs) == 3

    def test_page_content_matches_chunk_text(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(2)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert docs[0].page_content == "Chunk 0 body text."
        assert docs[1].page_content == "Chunk 1 body text."

    def test_metadata_source(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(1)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert docs[0].metadata["source"] == "paper.pdf"

    def test_metadata_section_path(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(1)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert docs[0].metadata["section_path"] == "Introduction"

    def test_metadata_page_numbers(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(1)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert docs[0].metadata["page_start"] == 1
        assert docs[0].metadata["page_end"] == 1

    def test_metadata_hierarchy_confidence(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(1, confidence=0.82)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert docs[0].metadata["hierarchy_confidence"] == pytest.approx(0.82)

    def test_metadata_chunk_links_present(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(2)):
            docs = list(DocuWeaveLoader("paper.pdf").lazy_load())
        assert "previous_chunk_id" in docs[0].metadata
        assert "next_chunk_id" in docs[0].metadata

    def test_lazy_load_returns_iterator(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(3)):
            result = DocuWeaveLoader("paper.pdf").lazy_load()
        assert hasattr(result, "__next__"), "lazy_load must return an iterator"

    def test_max_tokens_forwarded_to_iter_chunks(self) -> None:
        mock_doc = _mock_doc(1)
        with patch("docuweave.parse", return_value=mock_doc):
            list(DocuWeaveLoader("paper.pdf", max_tokens=256).lazy_load())
        mock_doc.iter_chunks.assert_called_once_with(max_tokens=256)

    def test_empty_pdf_yields_nothing(self) -> None:
        mock_doc = MagicMock()
        mock_doc.hierarchy_confidence = 0.0
        mock_doc.iter_chunks.return_value = iter([])
        with patch("docuweave.parse", return_value=mock_doc):
            docs = list(DocuWeaveLoader("empty.pdf").lazy_load())
        assert docs == []


class TestDocuWeaveLoaderLoad:
    def test_load_returns_list(self) -> None:
        with patch("docuweave.parse", return_value=_mock_doc(2)):
            docs = DocuWeaveLoader("paper.pdf").load()
        assert isinstance(docs, list)
        assert len(docs) == 2


class TestDocuWeaveLoaderErrors:
    def test_missing_docuweave_raises_import_error(self) -> None:
        with patch.dict("sys.modules", {"docuweave": None}):
            with pytest.raises(ImportError, match="docuweave"):
                list(DocuWeaveLoader("paper.pdf").lazy_load())

    def test_invalid_pdf_propagates_value_error(self) -> None:
        with patch("docuweave.parse", side_effect=ValueError("not a valid PDF")):
            with pytest.raises(ValueError):
                list(DocuWeaveLoader("bad.pdf").lazy_load())
