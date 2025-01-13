import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.audio import (
        OpenAIWhisperParser,
    )
    from langchain_community.document_loaders.parsers.doc_intelligence import (
        AzureAIDocumentIntelligenceParser,
    )
    from langchain_community.document_loaders.parsers.docai import (
        DocAIParser,
    )
    from langchain_community.document_loaders.parsers.grobid import (
        GrobidParser,
    )
    from langchain_community.document_loaders.parsers.html import (
        BS4HTMLParser,
    )
    from langchain_community.document_loaders.parsers.images import (
        BaseImageBlobParser,
        LLMImageBlobParser,
        RapidOCRBlobParser,
        TesseractBlobParser,
    )
    from langchain_community.document_loaders.parsers.language import (
        LanguageParser,
    )
    from langchain_community.document_loaders.parsers.pdf import (
        PDFMinerParser,
        PDFPlumberParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
    )
    from langchain_community.document_loaders.parsers.vsdx import (
        VsdxParser,
    )


_module_lookup = {
    "AzureAIDocumentIntelligenceParser": "langchain_community.document_loaders.parsers.doc_intelligence",  # noqa: E501
    "BS4HTMLParser": "langchain_community.document_loaders.parsers.html",
    "BaseImageBlobParser": "langchain_community.document_loaders.parsers.images",
    "DocAIParser": "langchain_community.document_loaders.parsers.docai",
    "GrobidParser": "langchain_community.document_loaders.parsers.grobid",
    "LanguageParser": "langchain_community.document_loaders.parsers.language",
    "LLMImageBlobParser": "langchain_community.document_loaders.parsers.images",
    "OpenAIWhisperParser": "langchain_community.document_loaders.parsers.audio",
    "PDFMinerParser": "langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "langchain_community.document_loaders.parsers.pdf",
    "RapidOCRBlobParser": "langchain_community.document_loaders.parsers.images",
    "TesseractBlobParser": "langchain_community.document_loaders.parsers.images",
    "VsdxParser": "langchain_community.document_loaders.parsers.vsdx",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "AzureAIDocumentIntelligenceParser",
    "BaseImageBlobParser",
    "BS4HTMLParser",
    "DocAIParser",
    "GrobidParser",
    "LanguageParser",
    "LLMImageBlobParser",
    "OpenAIWhisperParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFParser",
    "PyPDFium2Parser",
    "RapidOCRBlobParser",
    "TesseractBlobParser",
    "VsdxParser",
]
