from typing import Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)


class AzureAIDocumentIntelligenceLoader(BaseLoader):
    """Loads a PDF with Azure Document Intelligence"""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        file_path: Optional[str] = None,
        url_path: Optional[str] = None,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
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
            Either file_path or url_path must be specified.
        url_path : Optional[str]
            The URL to the file that needs to be loaded.
            Either file_path or url_path must be specified.
        api_version: Optional[str]
            The API version for DocumentIntelligenceClient. Setting None to use
            the default value from SDK.
        api_model: str
            The model name or ID to be used for form recognition in Azure.

        Examples:
        ---------
        >>> obj = AzureAIDocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     api_endpoint="https://endpoint.azure.com",
        ...     api_key="APIKEY",
        ...     api_version="2023-10-31-preview",
        ...     model="prebuilt-document"
        ... )
        """

        assert (
            file_path is not None or url_path is not None
        ), "file_path or url_path must be provided"
        self.file_path = file_path
        self.url_path = url_path

        self.parser = AzureAIDocumentIntelligenceParser(
            api_endpoint=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            api_model=api_model,
            mode=mode,
        )

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.file_path is not None:
            blob = Blob.from_path(self.file_path)
            yield from self.parser.parse(blob)
        else:
            yield from self.parser.parse_url(self.url_path)
