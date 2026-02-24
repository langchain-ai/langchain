"""Backwards-compatible re-export of ``BaseDocumentCompressor``.

Third-party packages such as ``ragatouille`` import from this path::

    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

This module ensures that import continues to work after the class was
moved to ``langchain_core.documents.compressor``.
"""

from langchain_core.documents.compressor import BaseDocumentCompressor

__all__ = ["BaseDocumentCompressor"]
