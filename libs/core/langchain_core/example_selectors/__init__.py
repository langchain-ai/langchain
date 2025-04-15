"""Example selectors.

**Example selector** implements logic for selecting examples to include them in prompts.
This allows us to select examples that are most relevant to the input.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.example_selectors.base import BaseExampleSelector
    from langchain_core.example_selectors.length_based import (
        LengthBasedExampleSelector,
    )
    from langchain_core.example_selectors.semantic_similarity import (
        MaxMarginalRelevanceExampleSelector,
        SemanticSimilarityExampleSelector,
        sorted_values,
    )

__all__ = [
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "SemanticSimilarityExampleSelector",
    "sorted_values",
]

_dynamic_imports = {
    "BaseExampleSelector": "base",
    "LengthBasedExampleSelector": "length_based",
    "MaxMarginalRelevanceExampleSelector": "semantic_similarity",
    "SemanticSimilarityExampleSelector": "semantic_similarity",
    "sorted_values": "semantic_similarity",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
