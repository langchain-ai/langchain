"""Embeddings."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings
    from langchain_core.embeddings.fake import (
        DeterministicFakeEmbedding,
        FakeEmbeddings,
    )

__all__ = ("DeterministicFakeEmbedding", "Embeddings", "FakeEmbeddings")

_dynamic_imports = {
    "Embeddings": "embeddings",
    "DeterministicFakeEmbedding": "fake",
    "FakeEmbeddings": "fake",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
