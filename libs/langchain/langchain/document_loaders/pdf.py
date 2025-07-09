from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        AmazonTextractPDFLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PagedPDFSplitter,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFium2Loader,
        UnstructuredPDFLoader,
    )
    from langchain_community.document_loaders.pdf import (
        BasePDFLoader,
        DocumentIntelligenceLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "UnstructuredPDFLoader": "langchain_community.document_loaders",
    "BasePDFLoader": "langchain_community.document_loaders.pdf",
    "OnlinePDFLoader": "langchain_community.document_loaders",
    "PagedPDFSplitter": "langchain_community.document_loaders",
    "PyPDFium2Loader": "langchain_community.document_loaders",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders",
    "PDFMinerLoader": "langchain_community.document_loaders",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders",
    "PyMuPDFLoader": "langchain_community.document_loaders",
    "MathpixPDFLoader": "langchain_community.document_loaders",
    "PDFPlumberLoader": "langchain_community.document_loaders",
    "AmazonTextractPDFLoader": "langchain_community.document_loaders",
    "DocumentIntelligenceLoader": "langchain_community.document_loaders.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AmazonTextractPDFLoader",
    "BasePDFLoader",
    "DocumentIntelligenceLoader",
    "MathpixPDFLoader",
    "OnlinePDFLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PDFPlumberLoader",
    "PagedPDFSplitter",
    "PyMuPDFLoader",
    "PyPDFDirectoryLoader",
    "PyPDFium2Loader",
    "UnstructuredPDFLoader",
]
