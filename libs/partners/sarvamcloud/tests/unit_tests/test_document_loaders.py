"""Unit tests for SarvamDocumentIntelligence."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestSarvamDocumentIntelligence:
    @pytest.fixture()
    def mock_sarvam_client(self) -> MagicMock:
        """Return a mock SarvamAI client."""
        client = MagicMock()
        client.document_intelligence.create_job.return_value = {
            "job_id": "doc-job-123",
            "job_state": "Accepted",
        }
        client.document_intelligence.get_upload_links.return_value = {
            "upload_urls": {
                "document.pdf": {
                    "url": "https://signed.example.com/upload",
                    "expires_in": 3600,
                }
            }
        }
        client.document_intelligence.get_status.return_value = {
            "job_id": "doc-job-123",
            "job_state": "Completed",
            "total_pages": 1,
            "successful_pages": 1,
            "failed_pages": 0,
        }
        client.document_intelligence.get_download_links.return_value = {
            "download_urls": {}
        }
        return client

    @pytest.fixture()
    def loader(self, mock_sarvam_client: MagicMock, tmp_path: "Path") -> "SarvamDocumentIntelligence":  # type: ignore[name-defined]
        from langchain_sarvamcloud.document_loaders import SarvamDocumentIntelligence

        # Create a temporary PDF file
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        with patch("sarvamai.SarvamAI", return_value=mock_sarvam_client):
            loader = SarvamDocumentIntelligence(
                file_paths=[str(pdf_file)],
                language="hi-IN",
                output_format="md",
                api_subscription_key="test-key",  # type: ignore[arg-type]
            )
        loader._client = mock_sarvam_client
        return loader

    def test_create_job_returns_job_id(
        self, loader: "SarvamDocumentIntelligence", mock_sarvam_client: MagicMock
    ) -> None:
        mock_sarvam_client.document_intelligence.create_job.return_value = {
            "job_id": "doc-job-999",
            "job_state": "Accepted",
        }
        job_id = loader._create_job()
        assert job_id == "doc-job-999"

    def test_poll_status_returns_completed(
        self, loader: "SarvamDocumentIntelligence", mock_sarvam_client: MagicMock
    ) -> None:
        mock_sarvam_client.document_intelligence.get_status.return_value = {
            "job_state": "Completed",
            "job_id": "abc",
        }
        status = loader._poll_status("abc")
        assert status["job_state"] == "Completed"

    def test_lazy_load_raises_on_failed_job(
        self, loader: "SarvamDocumentIntelligence", mock_sarvam_client: MagicMock
    ) -> None:
        mock_sarvam_client.document_intelligence.get_status.return_value = {
            "job_state": "Failed",
            "job_id": "doc-job-123",
        }
        mock_upload_client = MagicMock()
        mock_upload_client.__enter__.return_value.put.return_value = MagicMock(
            raise_for_status=lambda: None
        )
        with patch("httpx.Client", return_value=mock_upload_client):
            with pytest.raises(RuntimeError, match="failed"):
                list(loader.lazy_load())

    def test_download_results_returns_documents(
        self, loader: "SarvamDocumentIntelligence"
    ) -> None:
        download_urls = {
            "output.md": {"url": "https://signed.example.com/download", "expires_in": 3600}
        }
        mock_response = MagicMock()
        mock_response.text = "# Hello\n\nThis is the extracted content."
        mock_response.raise_for_status = lambda: None

        with patch("httpx.Client") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value.get.return_value = (
                mock_response
            )
            docs = loader._download_results(download_urls)

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Hello" in docs[0].page_content
        assert docs[0].metadata["source"] == "output.md"
        assert docs[0].metadata["language"] == "hi-IN"

    def test_default_language_is_hi_in(self) -> None:
        from langchain_sarvamcloud.document_loaders import SarvamDocumentIntelligence

        with patch("sarvamai.SarvamAI"):
            loader = SarvamDocumentIntelligence(
                file_paths=["doc.pdf"],
                api_subscription_key="key",  # type: ignore[arg-type]
            )
        assert loader.language == "hi-IN"

    def test_default_output_format_is_md(self) -> None:
        from langchain_sarvamcloud.document_loaders import SarvamDocumentIntelligence

        with patch("sarvamai.SarvamAI"):
            loader = SarvamDocumentIntelligence(
                file_paths=["doc.pdf"],
                api_subscription_key="key",  # type: ignore[arg-type]
            )
        assert loader.output_format == "md"
