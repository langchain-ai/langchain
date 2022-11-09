<<<<<<< HEAD
"""Wrappers on top of vectorstores."""
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
=======
"""Wrappers on top of vector stores."""
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores.faiss import FAISS
>>>>>>> master

__all__ = ["ElasticVectorSearch", "FAISS", "VectorStore"]
