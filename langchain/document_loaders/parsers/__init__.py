from langchain.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain.document_loaders.parsers.grobid import GrobidParser
from langchain.document_loaders.parsers.html import BS4HTMLParser
from langchain.document_loaders.parsers.language import LanguageParser
from langchain.document_loaders.parsers.text_extract import DoctranExtractParser
from langchain.document_loaders.parsers.text_qa import DoctranQAParser
from langchain.document_loaders.parsers.text_translate import DoctranTranslateParser
from langchain.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
)

__all__ = [
    "BS4HTMLParser",
    "GrobidParser",
    "LanguageParser",
    "OpenAIWhisperParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFium2Parser",
    "PyPDFParser",
    "DoctranExtractParser",
    "DoctranQAParser",
    "DoctranTranslateParser",
]
