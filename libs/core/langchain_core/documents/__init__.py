"""Documents module.

**Document** module is a collection of classes that handle documents
and their transformations.

"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Document
    from .compressor import BaseDocumentCompressor
    from .transformers import BaseDocumentTransformer

__all__ = ["Document", "BaseDocumentTransformer", "BaseDocumentCompressor"]

_dynamic_imports = {
    "Document": "base",
    "BaseDocumentCompressor": "compressor",
    "BaseDocumentTransformer": "transformers",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        try:
            result = import_module(f".{attr_name}", package=package)
        except ModuleNotFoundError:
            raise AttributeError(f"module '.' has no attribute {attr_name!r}")
    else:
        try:
            module = import_module(f".{module_name}", package=package)
        except ModuleNotFoundError:
            raise AttributeError(f"module '{module_name}' has no attribute {attr_name!r}")
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
