"""Sarvam AI Document Intelligence loader."""

from __future__ import annotations

import time
from typing import Any, Iterator, Literal

from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

_OUTPUT_FORMATS = Literal["md", "html", "json"]
_JOB_STATES = Literal["Accepted", "Pending", "Running", "Completed", "Failed"]


class SarvamDocumentIntelligence(BaseModel, BaseLoader):
    """Sarvam AI Document Intelligence loader.

    Digitizes PDF documents and image archives using Sarvam's Vision model
    (3B VLM). Supports OCR across 23 languages with table extraction, layout
    preservation, and reading order detection.

    This loader uses an async job workflow:
    1. Create a job.
    2. Get signed upload URLs.
    3. Upload files to the signed URLs.
    4. Start the job.
    5. Poll status until completed.
    6. Download and parse results as LangChain `Document` objects.

    Input: PDF files, PNG/JPEG images, or ZIP archives (flat structure).
        Max 500 pages per job, max 200 MB per file.

    Supported languages (23): All 22 official Indian languages + English.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Example:
        ```python
        from langchain_sarvamcloud import SarvamDocumentIntelligence

        loader = SarvamDocumentIntelligence(
            file_paths=["document.pdf"],
            language="hi-IN",
            output_format="md",
        )
        documents = loader.load()
        for doc in documents:
            print(doc.page_content[:200])
        ```
    """

    file_paths: list[str]
    """Paths to local files to digitize (PDF, PNG, JPEG, or ZIP)."""

    language: str = "hi-IN"
    """BCP-47 language code for OCR (e.g. `hi-IN`, `en-IN`)."""

    output_format: _OUTPUT_FORMATS = "md"
    """Output format for extracted content.

    - `md`: Markdown (default).
    - `html`: HTML with layout preservation.
    - `json`: Structured JSON for programmatic processing.
    """

    callback: str | None = None
    """Optional webhook URL for job completion notifications."""

    poll_interval: float = 5.0
    """Seconds between status poll requests."""

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def _create_job(self) -> str:
        """Create a document intelligence job.

        Returns:
            The `job_id` string.
        """
        kwargs: dict[str, Any] = {
            "language": self.language,
            "output_format": self.output_format,
        }
        if self.callback:
            kwargs["callback_url"] = self.callback
        response = self._client.document_intelligence.create_job(**kwargs)
        if not isinstance(response, dict):
            response = response.model_dump()
        return response["job_id"]

    def _get_upload_urls(
        self, job_id: str, filenames: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get signed upload URLs for files.

        Args:
            job_id: The job ID.
            filenames: List of file names.

        Returns:
            Dict mapping filenames to upload URL info.
        """
        response = self._client.document_intelligence.get_upload_links(
            job_id=job_id, files=filenames
        )
        if not isinstance(response, dict):
            response = response.model_dump()
        return response.get("upload_urls", {})

    def _upload_files(
        self,
        file_paths: list[str],
        upload_urls: dict[str, dict[str, Any]],
    ) -> None:
        """Upload files to their signed URLs.

        Args:
            file_paths: Local paths to the files.
            upload_urls: Map of filename → upload URL info.
        """
        import os  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        with httpx.Client() as client:
            for path in file_paths:
                filename = os.path.basename(path)
                url_info = upload_urls.get(filename)
                if not url_info:
                    msg = f"No upload URL received for file: {filename}"
                    raise ValueError(msg)
                with open(path, "rb") as f:
                    response = client.put(url_info["url"], content=f.read())
                    response.raise_for_status()

    def _start_job(self, job_id: str) -> None:
        """Start processing the job.

        Args:
            job_id: The job ID.
        """
        self._client.document_intelligence.start(job_id=job_id)

    def _poll_status(self, job_id: str) -> dict[str, Any]:
        """Poll until the job reaches a terminal state.

        Args:
            job_id: The job ID.

        Returns:
            Final status dict with `job_state`, `total_pages`, etc.
        """
        while True:
            response = self._client.document_intelligence.get_status(job_id=job_id)
            if not isinstance(response, dict):
                response = response.model_dump()
            if response["job_state"] in ("Completed", "Failed"):
                return response
            time.sleep(self.poll_interval)

    def _get_download_urls(self, job_id: str) -> dict[str, dict[str, Any]]:
        """Get signed download URLs for the job output.

        Args:
            job_id: The job ID.

        Returns:
            Dict mapping output filenames to download URL info.
        """
        response = self._client.document_intelligence.get_download_links(
            job_id=job_id
        )
        if not isinstance(response, dict):
            response = response.model_dump()
        return response.get("download_urls", {})

    def _download_results(
        self, download_urls: dict[str, dict[str, Any]]
    ) -> list[Document]:
        """Download and parse output files into Document objects.

        Args:
            download_urls: Map of output filename → download URL info.

        Returns:
            List of LangChain `Document` objects with extracted text.
        """
        import io  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        documents = []
        with httpx.Client() as client:
            for filename, url_info in download_urls.items():
                response = client.get(url_info["url"])
                response.raise_for_status()
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        for name in zf.namelist():
                            content = zf.read(name).decode("utf-8", errors="replace")
                            documents.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": name,
                                        "format": self.output_format,
                                        "language": self.language,
                                    },
                                )
                            )
                else:
                    content = response.text
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": filename,
                                "format": self.output_format,
                                "language": self.language,
                            },
                        )
                    )
        return documents

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily via the Sarvam Document Intelligence API.

        Runs the full async job workflow: create → upload → start → poll →
        download. Each output file becomes one or more `Document` objects.

        Yields:
            `Document` objects with extracted text content and metadata.

        Raises:
            RuntimeError: If the digitization job fails.
        """
        import os  # noqa: PLC0415

        filenames = [os.path.basename(p) for p in self.file_paths]

        job_id = self._create_job()
        upload_urls = self._get_upload_urls(job_id, filenames)
        self._upload_files(self.file_paths, upload_urls)
        self._start_job(job_id)
        status = self._poll_status(job_id)

        if status["job_state"] == "Failed":
            msg = f"Sarvam Document Intelligence job {job_id} failed: {status}"
            raise RuntimeError(msg)

        download_urls = self._get_download_urls(job_id)
        yield from self._download_results(download_urls)
