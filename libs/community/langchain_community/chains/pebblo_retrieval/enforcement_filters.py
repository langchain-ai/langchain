"""
Identity & Semantic Enforcement filters for PebbloRetrievalQA chain:

This module contains methods for applying Identity and Semantic Enforcement filters
in the PebbloRetrievalQA chain.
These filters are used to control the retrieval of documents based on authorization and
semantic context.
The Identity Enforcement filter ensures that only authorized identities can access
certain documents, while the Semantic Enforcement filter controls document retrieval
based on semantic context.

The methods in this module are designed to work with different types of vector stores.
"""

import logging
from typing import List, Optional, Union

from langchain_core.vectorstores import VectorStoreRetriever

from langchain_community.chains.pebblo_retrieval.models import (
    AuthContext,
    SemanticContext,
)
from langchain_community.vectorstores import Pinecone, Qdrant

logger = logging.getLogger(__name__)

SUPPORTED_VECTORSTORES = [Pinecone, Qdrant]


def set_enforcement_filters(
    retriever: VectorStoreRetriever,
    auth_context: Optional[AuthContext],
    semantic_context: Optional[SemanticContext],
) -> None:
    """
    Set identity and semantic enforcement filters in the retriever.
    """
    if auth_context is not None:
        _set_identity_enforcement_filter(retriever, auth_context)
    if semantic_context is not None:
        _set_semantic_enforcement_filter(retriever, semantic_context)


def _apply_qdrant_semantic_filter(
    search_kwargs: dict, semantic_context: Optional[SemanticContext]
) -> None:
    """
    Set semantic enforcement filter in search_kwargs for Qdrant vectorstore.
    """
    try:
        from qdrant_client.http import models as rest
    except ImportError as e:
        raise ValueError(
            "Could not import `qdrant-client.http` python package. "
            "Please install it with `pip install qdrant-client`."
        ) from e

    # Create a semantic enforcement filter condition
    semantic_filters: List[
        Union[
            rest.FieldCondition,
            rest.IsEmptyCondition,
            rest.IsNullCondition,
            rest.HasIdCondition,
            rest.NestedCondition,
            rest.Filter,
        ]
    ] = []

    if (
        semantic_context is not None
        and semantic_context.pebblo_semantic_topics is not None
    ):
        semantic_topics_filter = rest.FieldCondition(
            key="metadata.pebblo_semantic_topics",
            match=rest.MatchAny(any=semantic_context.pebblo_semantic_topics.deny),
        )
        semantic_filters.append(semantic_topics_filter)
    if (
        semantic_context is not None
        and semantic_context.pebblo_semantic_entities is not None
    ):
        semantic_entities_filter = rest.FieldCondition(
            key="metadata.pebblo_semantic_entities",
            match=rest.MatchAny(any=semantic_context.pebblo_semantic_entities.deny),
        )
        semantic_filters.append(semantic_entities_filter)

    # If 'filter' already exists in search_kwargs
    if "filter" in search_kwargs:
        existing_filter: rest.Filter = search_kwargs["filter"]

        # Check if existing_filter is a qdrant-client filter
        if isinstance(existing_filter, rest.Filter):
            # If 'must_not' condition exists in the existing filter
            if isinstance(existing_filter.must_not, list):
                # Warn if 'pebblo_semantic_topics' or 'pebblo_semantic_entities'
                # filter is overridden
                new_must_not_conditions: List[
                    Union[
                        rest.FieldCondition,
                        rest.IsEmptyCondition,
                        rest.IsNullCondition,
                        rest.HasIdCondition,
                        rest.NestedCondition,
                        rest.Filter,
                    ]
                ] = []
                # Drop semantic filter conditions if already present
                for condition in existing_filter.must_not:
                    if hasattr(condition, "key"):
                        if condition.key == "metadata.pebblo_semantic_topics":
                            continue
                        if condition.key == "metadata.pebblo_semantic_entities":
                            continue
                        new_must_not_conditions.append(condition)
                # Add semantic enforcement filters to 'must_not' conditions
                existing_filter.must_not = new_must_not_conditions
                existing_filter.must_not.extend(semantic_filters)
            else:
                # Set 'must_not' condition with semantic enforcement filters
                existing_filter.must_not = semantic_filters
        else:
            raise TypeError(
                "Using dict as a `filter` is deprecated. "
                "Please use qdrant-client filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/"
            )
    else:
        # If 'filter' does not exist in search_kwargs, create it
        search_kwargs["filter"] = rest.Filter(must_not=semantic_filters)


def _apply_qdrant_authorization_filter(
    search_kwargs: dict, auth_context: Optional[AuthContext]
) -> None:
    """
    Set identity enforcement filter in search_kwargs for Qdrant vectorstore.
    """
    try:
        from qdrant_client.http import models as rest
    except ImportError as e:
        raise ValueError(
            "Could not import `qdrant-client.http` python package. "
            "Please install it with `pip install qdrant-client`."
        ) from e

    if auth_context is not None:
        # Create a identity enforcement filter condition
        identity_enforcement_filter = rest.FieldCondition(
            key="metadata.authorized_identities",
            match=rest.MatchAny(any=auth_context.user_auth),
        )
    else:
        return

    # If 'filter' already exists in search_kwargs
    if "filter" in search_kwargs:
        existing_filter: rest.Filter = search_kwargs["filter"]

        # Check if existing_filter is a qdrant-client filter
        if isinstance(existing_filter, rest.Filter):
            # If 'must' exists in the existing filter
            if existing_filter.must:
                new_must_conditions: List[
                    Union[
                        rest.FieldCondition,
                        rest.IsEmptyCondition,
                        rest.IsNullCondition,
                        rest.HasIdCondition,
                        rest.NestedCondition,
                        rest.Filter,
                    ]
                ] = []
                # Drop 'authorized_identities' filter condition if already present
                for condition in existing_filter.must:
                    if (
                        hasattr(condition, "key")
                        and condition.key == "metadata.authorized_identities"
                    ):
                        continue
                    new_must_conditions.append(condition)

                # Add identity enforcement filter to 'must' conditions
                existing_filter.must = new_must_conditions
                existing_filter.must.append(identity_enforcement_filter)
            else:
                # Set 'must' condition with identity enforcement filter
                existing_filter.must = [identity_enforcement_filter]
        else:
            raise TypeError(
                "Using dict as a `filter` is deprecated. "
                "Please use qdrant-client filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/"
            )
    else:
        # If 'filter' does not exist in search_kwargs, create it
        search_kwargs["filter"] = rest.Filter(must=[identity_enforcement_filter])


def _apply_pinecone_semantic_filter(
    search_kwargs: dict, semantic_context: Optional[SemanticContext]
) -> None:
    """
    Set semantic enforcement filter in search_kwargs for Pinecone vectorstore.
    """
    # Check if semantic_context is provided
    semantic_context = semantic_context
    if semantic_context is not None:
        if semantic_context.pebblo_semantic_topics is not None:
            # Add pebblo_semantic_topics filter to search_kwargs
            search_kwargs.setdefault("filter", {})["pebblo_semantic_topics"] = {
                "$nin": semantic_context.pebblo_semantic_topics.deny
            }

        if semantic_context.pebblo_semantic_entities is not None:
            # Add pebblo_semantic_entities filter to search_kwargs
            search_kwargs.setdefault("filter", {})["pebblo_semantic_entities"] = {
                "$nin": semantic_context.pebblo_semantic_entities.deny
            }


def _apply_pinecone_authorization_filter(
    search_kwargs: dict, auth_context: Optional[AuthContext]
) -> None:
    """
    Set identity enforcement filter in search_kwargs for Pinecone vectorstore.
    """
    if auth_context is not None:
        search_kwargs.setdefault("filter", {})["authorized_identities"] = {
            "$in": auth_context.user_auth
        }


def _set_identity_enforcement_filter(
    retriever: VectorStoreRetriever, auth_context: Optional[AuthContext]
) -> None:
    """
    Set identity enforcement filter in search_kwargs.

    This method sets the identity enforcement filter in the search_kwargs
    of the retriever based on the type of the vectorstore.
    """
    search_kwargs = retriever.search_kwargs
    if isinstance(retriever.vectorstore, Pinecone):
        _apply_pinecone_authorization_filter(search_kwargs, auth_context)
    elif isinstance(retriever.vectorstore, Qdrant):
        _apply_qdrant_authorization_filter(search_kwargs, auth_context)


def _set_semantic_enforcement_filter(
    retriever: VectorStoreRetriever, semantic_context: Optional[SemanticContext]
) -> None:
    """
    Set semantic enforcement filter in search_kwargs.

    This method sets the semantic enforcement filter in the search_kwargs
    of the retriever based on the type of the vectorstore.
    """
    search_kwargs = retriever.search_kwargs
    if isinstance(retriever.vectorstore, Pinecone):
        _apply_pinecone_semantic_filter(search_kwargs, semantic_context)
    elif isinstance(retriever.vectorstore, Qdrant):
        _apply_qdrant_semantic_filter(search_kwargs, semantic_context)
