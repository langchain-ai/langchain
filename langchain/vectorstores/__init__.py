"""Wrappers on top of vector stores."""
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores.faiss import FAISS

__all__ = ["ElasticVectorSearch", "FAISS", "VectorStore"]
