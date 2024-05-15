"""
Chains module for langchain_community

This module contains the community chains.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.chains.pebblo_retrieval.base import PebbloRetrievalQA

__all__ = ["PebbloRetrievalQA"]

_module_lookup = {
    "PebbloRetrievalQA": "langchain_community.chains.pebblo_retrieval.base"
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
