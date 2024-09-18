from typing import Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)


class AzureAIDocumentIntelligenceLoader(BaseLoader):
    """Load a PDF with Azure Document Intelligence."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        file_path: Optional[str] = None,
        url_path: Optional[str] = None,
        bytes_source: Optional[bytes] = None,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        *,
        analysis_features: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a AzureAIDocumentIntelligenceParser object to be
        used for parsing files using the Azure Document Intelligence API. The load
        method generates Documents whose content representations are determined by the
        mode parameter.

        Parameters:
        -----------
        api_endpoint: str
            The API endpoint to use for DocumentIntelligenceClient construction.
        api_key: str
            The API key to use for DocumentIntelligenceClient construction.
        file_path : Optional[str]
            The path to the file that needs to be loaded.
            Either file_path, url_path or bytes_source must be specified.
        url_path : Optional[str]
            The URL to the file that needs to be loaded.
            Either file_path, url_path or bytes_source must be specified.
        bytes_source : Optional[bytes]
            The bytes array of the file that needs to be loaded.
            Either file_path, url_path or bytes_source must be specified.
        api_version: Optional[str]
            The API version for DocumentIntelligenceClient. Setting None to use
            the default value from `azure-ai-documentintelligence` package.
        api_model: str
            Unique document model name. Default value is "prebuilt-layout".
            Note that overriding this default value may result in unsupported
            behavior.
        mode: Optional[str]
            The type of content representation of the generated Documents.
            Use either "single", "page", or "markdown". Default value is "markdown".
        analysis_features: Optional[List[str]]
            List of optional analysis features, each feature should be passed
            as a str that conforms to the enum `DocumentAnalysisFeature` in
            `azure-ai-documentintelligence` package. Default value is None.

        Examples:
        ---------
        >>> obj = AzureAIDocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     api_endpoint="https://endpoint.azure.com",
        ...     api_key="APIKEY",
        ...     api_version="2023-10-31-preview",
        ...     api_model="prebuilt-layout",
        ...     mode="markdown"
        ... )
        """

        assert (
            file_path is not None or url_path is not None or bytes_source is not None
        ), "file_path, url_path or bytes_source must be provided"
        self.file_path = file_path
        self.url_path = url_path
        self.bytes_source = bytes_source

        self.parser = AzureAIDocumentIntelligenceParser(  # type: ignore[misc]
            api_endpoint=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            api_model=api_model,
            mode=mode,
            analysis_features=analysis_features,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load the document as pages."""
        if self.file_path is not None:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
            yield from self.parser.parse(blob)
        elif self.url_path is not None:
            yield from self.parser.parse_url(self.url_path)  # type: ignore[arg-type]
        elif self.bytes_source is not None:
            yield from self.parser.parse_bytes(self.bytes_source)
        else:
            raise ValueError("No data source provided.")
