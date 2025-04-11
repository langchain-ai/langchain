"""Documents module.

**Document** module is a collection of classes that handle documents
and their transformations.

"""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from .base import Document
    from .compressor import BaseDocumentCompressor
    from .transformers import BaseDocumentTransformer

__all__ = ["Document", "BaseDocumentTransformer", "BaseDocumentCompressor"]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="documents",
    dynamic_imports={
        "Document": "base",
        "BaseDocumentCompressor": "compressor",
        "BaseDocumentTransformer": "transformers",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
