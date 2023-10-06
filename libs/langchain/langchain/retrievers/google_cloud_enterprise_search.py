"""Retriever wrapper for Google Vertex AI Search.
    DEPRECATED: Maintained for backwards compatibility.
"""
from typing import Any

from langchain.retrievers.google_vertex_ai_search import GoogleVertexAISearchRetriever


class GoogleCloudEnterpriseSearchRetriever(GoogleVertexAISearchRetriever):
    """`Google Vertex Search API` retriever alias for backwards compatibility.
    DEPRECATED: Use `GoogleVertexAISearchRetriever` instead.
    """

    def __init__(self, **data: Any):
        import warnings

        warnings.warn(
            "GoogleCloudEnterpriseSearchRetriever is deprecated, use GoogleVertexAISearchRetriever",  # noqa: E501
            DeprecationWarning,
        )

        super().__init__(**data)
