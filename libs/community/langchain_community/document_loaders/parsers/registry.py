"""Module includes a registry of default parser configurations."""

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
from langchain_community.document_loaders.parsers.txt import TextParser


def _get_default_parser() -> BaseBlobParser:
    """Get default mime-type based parser."""
    return MimeTypeBasedParser(
        handlers={
            "application/pdf": PyMuPDFParser(),
            "text/plain": TextParser(),
            "application/msword": MsWordParser(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
                MsWordParser()
            ),
        },
        fallback_parser=None,
    )


_REGISTRY = {
    "default": _get_default_parser,
}

# PUBLIC API


def get_parser(parser_name: str) -> BaseBlobParser:
    """Get a parser by parser name."""
    if parser_name not in _REGISTRY:
        raise ValueError(f"Unknown parser combination: {parser_name}")
    return _REGISTRY[parser_name]()
