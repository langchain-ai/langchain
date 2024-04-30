from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.pdf import (
        AmazonTextractPDFLoader,
        BasePDFLoader,
        DocumentIntelligenceLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFium2Loader,
        PyPDFLoader,
        UnstructuredPDFLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "UnstructuredPDFLoader": "langchain_community.document_loaders.pdf",
    "BasePDFLoader": "langchain_community.document_loaders.pdf",
    "OnlinePDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFium2Loader": "langchain_community.document_loaders.pdf",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders.pdf",
    "PyMuPDFLoader": "langchain_community.document_loaders.pdf",
    "MathpixPDFLoader": "langchain_community.document_loaders.pdf",
    "PDFPlumberLoader": "langchain_community.document_loaders.pdf",
    "AmazonTextractPDFLoader": "langchain_community.document_loaders.pdf",
    "DocumentIntelligenceLoader": "langchain_community.document_loaders.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "UnstructuredPDFLoader",
    "BasePDFLoader",
    "OnlinePDFLoader",
    "PyPDFLoader",
    "PyPDFium2Loader",
    "PyPDFDirectoryLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PyMuPDFLoader",
    "MathpixPDFLoader",
    "PDFPlumberLoader",
    "AmazonTextractPDFLoader",
    "DocumentIntelligenceLoader",
]
