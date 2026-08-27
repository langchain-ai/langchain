from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.pdf import (
        AmazonTextractPDFParser,
        DocumentIntelligenceParser,
        PDFMinerParser,
        PDFPlumberParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
        extract_from_images_with_rapidocr,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "extract_from_images_with_rapidocr": (
        "langchain_community.document_loaders.parsers.pdf"
    ),
    "PyPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PDFMinerParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "langchain_community.document_loaders.parsers.pdf",
    "AmazonTextractPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "DocumentIntelligenceParser": "langchain_community.document_loaders.parsers.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AmazonTextractPDFParser",
    "DocumentIntelligenceParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFParser",
    "PyPDFium2Parser",
    "extract_from_images_with_rapidocr",
]
