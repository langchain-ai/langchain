from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

logger = logging.getLogger(__name__)


class AzureAIDocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer)."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        analysis_features: Optional[List[str]] = None,
        azure_credential: Optional["TokenCredential"] = None,
    ):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import DocumentAnalysisFeature
        from azure.core.credentials import AzureKeyCredential

        kwargs = {}

        credential: Union[AzureKeyCredential, TokenCredential]
        if azure_credential:
            if api_key is not None:
                raise ValueError(
                    "Only one of api_key or azure_credential should be provided."
                )
            credential = azure_credential
        elif api_key is not None:
            credential = AzureKeyCredential(api_key)
        else:
            raise ValueError("Either api_key or azure_credential must be provided.")

        if api_version is not None:
            kwargs["api_version"] = api_version

        self.client = DocumentIntelligenceClient(
            endpoint=api_endpoint,
            credential=credential,
            headers={"x-ms-useragent": "langchain-parser/1.0.0"},
            **kwargs,
        )
        self.api_model = api_model
        self.mode = mode
        self.features: Optional[List[DocumentAnalysisFeature]] = None
        if analysis_features is not None:
            self.features = [
                DocumentAnalysisFeature(feature) for feature in analysis_features
            ]
        assert self.mode in ["single", "page", "markdown"]

    def _generate_docs_page(self, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "page": p.page_number,
                },
            )
            yield d

    def _generate_docs_single(self, result: Any) -> Iterator[Document]:
        yield Document(page_content=result.content, metadata=result.as_dict())

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(
                self.api_model,
                body=file_obj,
                content_type="application/octet-stream",
                output_content_format="markdown" if self.mode == "markdown" else "text",
                features=self.features,
            )
            result = poller.result()

            if self.mode in ["single", "markdown"]:
                yield from self._generate_docs_single(result)
            elif self.mode in ["page"]:
                yield from self._generate_docs_page(result)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

    def parse_url(self, url: str) -> Iterator[Document]:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        poller = self.client.begin_analyze_document(
            self.api_model,
            body=AnalyzeDocumentRequest(url_source=url),
            output_content_format="markdown" if self.mode == "markdown" else "text",
            features=self.features,
        )
        result = poller.result()

        if self.mode in ["single", "markdown"]:
            yield from self._generate_docs_single(result)
        elif self.mode in ["page"]:
            yield from self._generate_docs_page(result)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def parse_bytes(self, bytes_source: bytes) -> Iterator[Document]:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        poller = self.client.begin_analyze_document(
            self.api_model,
            body=AnalyzeDocumentRequest(bytes_source=bytes_source),
            output_content_format="markdown" if self.mode == "markdown" else "text",
            features=self.features,
        )
        result = poller.result()

        if self.mode in ["single", "markdown"]:
            yield from self._generate_docs_single(result)
        elif self.mode in ["page"]:
            yield from self._generate_docs_page(result)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
