"""Embeddings."""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings
    from langchain_core.embeddings.fake import (
        DeterministicFakeEmbedding,
        FakeEmbeddings,
    )

__all__ = ["DeterministicFakeEmbedding", "Embeddings", "FakeEmbeddings"]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="embeddings",
    dynamic_imports={
        "Embeddings": "embeddings",
        "DeterministicFakeEmbedding": "fake",
        "FakeEmbeddings": "fake",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
