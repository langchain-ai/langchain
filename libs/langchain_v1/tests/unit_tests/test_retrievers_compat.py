"""Test backwards-compatible retriever imports.

These tests verify that the import paths used by third-party packages
(e.g., ``ragatouille``) continue to work after the module was moved
to ``langchain-core``.

See: https://github.com/langchain-ai/langchain/issues/35405
"""

from langchain_core.documents.compressor import (
    BaseDocumentCompressor as CoreBaseDocumentCompressor,
)
from langchain_core.retrievers import BaseRetriever as CoreBaseRetriever

from langchain.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import (
    BaseDocumentCompressor as CompressorFromPkg,
)
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor as CompressorFromBase,
)


def test_base_retriever_import() -> None:
    """Test that BaseRetriever can be imported from langchain.retrievers."""
    assert BaseRetriever is CoreBaseRetriever


def test_base_document_compressor_from_package() -> None:
    """Test that BaseDocumentCompressor can be imported from the package init."""
    assert CompressorFromPkg is CoreBaseDocumentCompressor


def test_base_document_compressor_from_base_module() -> None:
    """Test the exact import path used by ragatouille.

    ``from langchain.retrievers.document_compressors.base import BaseDocumentCompressor``
    """
    assert CompressorFromBase is CoreBaseDocumentCompressor
