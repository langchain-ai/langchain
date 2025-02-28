"""Documents module.

**Document** module is a collection of classes that handle documents
and their transformations.

"""

from langchain_core.documents.base import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.documents.transformers import BaseDocumentTransformer

__all__ = ["Document", "BaseDocumentTransformer", "BaseDocumentCompressor"]
