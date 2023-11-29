from typing import Iterator, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import DocumentIntelligenceParser


class DocumentIntelligenceLoader(BaseLoader):
    """Loads a PDF with Azure Document Intelligence"""

    def __init__(
        self,
        file_path: str,
        api_endpoint: str,
        api_key: str,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
    ) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a DocumentIntelligenceParser object to be used
        for parsing files using the Azure Document Intelligence API. The load method
        generates Documents whose content representations are determined by the mode
        parameter.

        Parameters:
        -----------
        file_path : str
            The path to the file that needs to be parsed.
        api_endpoint: str
            The API endpoint to use for DocumentIntelligenceClient construction.
        api_key: str
            The API key to use for DocumentIntelligenceClient construction.
        api_version: Optional[str]
            The API version for DocumentIntelligenceClient. Setting None to use
            the default value from SDK.
        api_model: str
            The model name or ID to be used for form recognition in Azure.

        Examples:
        ---------
        >>> obj = DocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     api_endpoint="https://endpoint.azure.com",
        ...     api_key="APIKEY",
        ...     api_version="2023-10-31-preview",
        ...     model="prebuilt-document"
        ... )
        """

        self.file_path = file_path

        self.parser = DocumentIntelligenceParser(
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
        blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)
