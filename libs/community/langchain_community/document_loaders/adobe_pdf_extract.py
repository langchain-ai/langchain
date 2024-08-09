from typing import Iterator

from langchain_core.documents import Document

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import (
    AdobePDFExtractParser,
)


class AdobePDFExtractLoader:
    """Loads a PDF with Adobe PDF Extraction."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        file_path: str,
        mode: str = "chunks",
        embed_figures: bool = True,
    ):
        """
        Initialize the object for file processing with Adobe PDF Extraction.

        This constructor initializes a AdobePDFExtractionParser object to be
        used for parsing files using the Adobe PDF Extraction API. The load
        method generates Documents whose content representations are determined by the
        mode parameter.

        Parameters:
        -----------
        client_id: str
            The API client id to use for ServicePrincipalCredentials construction.
        client_secret: str
            The API client secret to use for ServicePrincipalCredentials construction.
        file_path : str
            The path to the file that needs to be loaded.
        mode: Optional[str]
            The type of content representation of the generated Documents.
            Use either "json", "chunks", or "data". Default value is "chunks".
        embed_figures: bool
            Whether to embed figures in the generated Documents. The figures
            will be represented as placeholders with their corresponding
            base64-encoded image data in the document metadata. Default value is True.

        Examples:
        ---------
        >>> obj = AdobePDFExtractionLoader(
        ...     client_id="CLIENTID",
        ...     client_secret="CLIENT
        ...     file_path="path/to/file",
        ...     mode="chunks"
        ...     embed_figures=True
        ... )
        """

        assert file_path is not None, "file_path must be provided"
        self.file_path = file_path

        self.parser = AdobePDFExtractParser(  # type: ignore[misc]
            client_id=client_id,
            client_secret=client_secret,
            mode=mode,
            embed_figures=embed_figures,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)
