"""Embeddings."""

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding, FakeEmbeddings

__all__ = ["DeterministicFakeEmbedding", "Embeddings", "FakeEmbeddings"]
