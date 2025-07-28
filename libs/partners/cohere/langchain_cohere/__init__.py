"""Cohere integration package for LangChain.

This package contains the Cohere integrations for LangChain.
For the full implementation, please install: pip install langchain-cohere

This is a placeholder package to enable API documentation generation.
"""

# Import from local modules for documentation generation
from .chat_models import ChatCohere
from .common import CohereCitation
from .embeddings import CohereEmbeddings
from .rag_retrievers import CohereRagRetriever
from .rerank import CohereRerank

__all__ = [
    "CohereCitation",
    "ChatCohere",
    "CohereEmbeddings",
    "CohereRagRetriever",
    "CohereRerank",
]