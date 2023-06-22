from langchain.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain.document_loaders.parsers.html import BS4HTMLParser
from langchain.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
)

__all__ = [
    "BS4HTMLParser",
    "OpenAIWhisperParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFium2Parser",
    "PyPDFParser",
]
