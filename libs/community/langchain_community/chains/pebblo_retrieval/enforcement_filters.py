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
from typing import Any, List, Optional, Union

from langchain_core.vectorstores import VectorStoreRetriever

from langchain_community.chains.pebblo_retrieval.models import (
    AuthContext,
    SemanticContext,
)

logger = logging.getLogger(__name__)

PINECONE = "Pinecone"
QDRANT = "Qdrant"
PGVECTOR = "PGVector"

SUPPORTED_VECTORSTORES = {PINECONE, QDRANT, PGVECTOR}


def clear_enforcement_filters(retriever: VectorStoreRetriever) -> None:
    """
    Clear the identity and semantic enforcement filters in the retriever search_kwargs.
    """
    if retriever.vectorstore.__class__.__name__ == PGVECTOR:
        search_kwargs = retriever.search_kwargs
        if "filter" in search_kwargs:
            filters = search_kwargs["filter"]
            _pgvector_clear_pebblo_filters(
                search_kwargs, filters, "authorized_identities"
            )
            _pgvector_clear_pebblo_filters(
                search_kwargs, filters, "pebblo_semantic_topics"
            )
            _pgvector_clear_pebblo_filters(
                search_kwargs, filters, "pebblo_semantic_entities"
            )


def set_enforcement_filters(
    retriever: VectorStoreRetriever,
    auth_context: Optional[AuthContext],
    semantic_context: Optional[SemanticContext],
) -> None:
    """
    Set identity and semantic enforcement filters in the retriever.
    """
    # Clear existing enforcement filters
    clear_enforcement_filters(retriever)
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


def _apply_pgvector_filter(
    search_kwargs: dict, filters: Optional[Any], pebblo_filter: dict
) -> None:
    """
    Apply pebblo filters in the search_kwargs filters.
    """
    if isinstance(filters, dict):
        if len(filters) == 1:
            # The only operators allowed at the top level are $and, $or, and $not
            # First check if an operator or a field
            key, value = list(filters.items())[0]
            if key.startswith("$"):
                # Then it's an operator
                if key.lower() not in ["$and", "$or", "$not"]:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not "
                        f"but got: {key}"
                    )
                if not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )

                # Here we handle the $and, $or, and $not operators(Semantic filters)
                if key.lower() == "$and":
                    # Add pebblo_filter to the $and list as it is
                    value.append(pebblo_filter)
                elif key.lower() == "$not":
                    # Check if pebblo_filter is an operator or a field
                    _key, _value = list(pebblo_filter.items())[0]
                    if _key.startswith("$"):
                        # Then it's a operator
                        if _key.lower() == "$not":
                            # It's Semantic filter, add it's value to filters
                            value.append(_value)
                            logger.warning(
                                "Adding $not operator to the existing $not operator"
                            )
                            return
                        else:
                            # Only $not operator is supported in pebblo_filter
                            raise ValueError(
                                f"Invalid filter key. Expected '$not' but got: {_key}"
                            )
                    else:
                        # Then it's a field(Auth filter), move filters into $and
                        search_kwargs["filter"] = {"$and": [filters, pebblo_filter]}
                        return
                elif key.lower() == "$or":
                    search_kwargs["filter"] = {"$and": [filters, pebblo_filter]}
            else:
                # Then it's a field and we can check pebblo_filter now
                # Check if pebblo_filter is an operator or a field
                _key, _ = list(pebblo_filter.items())[0]
                if _key.startswith("$"):
                    # Then it's a operator
                    if _key.lower() == "$not":
                        # It's a $not operator(Semantic filter), move filters into $and
                        search_kwargs["filter"] = {"$and": [filters, pebblo_filter]}
                        return
                    else:
                        # Only $not operator is allowed in pebblo_filter
                        raise ValueError(
                            f"Invalid filter key. Expected '$not' but got: {_key}"
                        )
                else:
                    # Then it's a field(This handles Auth filter)
                    filters.update(pebblo_filter)
                    return
        elif len(filters) > 1:
            # Then all keys have to be fields (they cannot be operators)
            for key in filters.keys():
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            # filters should all be fields and we can check pebblo_filter now
            # Check if pebblo_filter is an operator or a field
            _key, _ = list(pebblo_filter.items())[0]
            if _key.startswith("$"):
                # Then it's a operator
                if _key.lower() == "$not":
                    # It's a $not operator(Semantic filter), move filters into '$and'
                    search_kwargs["filter"] = {"$and": [filters, pebblo_filter]}
                    return
                else:
                    # Only $not operator is supported in pebblo_filter
                    raise ValueError(
                        f"Invalid filter key. Expected '$not' but got: {_key}"
                    )
            else:
                # Then it's a field(This handles Auth filter)
                filters.update(pebblo_filter)
                return
        else:
            # Got an empty dictionary for filters, set pebblo_filter in filter
            search_kwargs.setdefault("filter", {}).update(pebblo_filter)
    elif filters is None:
        # If filters is None, set pebblo_filter as a new filter
        search_kwargs.setdefault("filter", {}).update(pebblo_filter)
    else:
        raise ValueError(
            f"Invalid filter. Expected a dictionary/None but got type: {type(filters)}"
        )


def _pgvector_clear_pebblo_filters(
    search_kwargs: dict, filters: dict, pebblo_filter_key: str
) -> None:
    """
    Remove pebblo filters from the search_kwargs filters.
    """
    if isinstance(filters, dict):
        if len(filters) == 1:
            # The only operators allowed at the top level are $and, $or, and $not
            # First check if an operator or a field
            key, value = list(filters.items())[0]
            if key.startswith("$"):
                # Then it's an operator
                # Validate the operator's key and value type
                if key.lower() not in ["$and", "$or", "$not"]:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not "
                        f"but got: {key}"
                    )
                elif not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )

                # Here we handle the $and, $or, and $not operators
                if key.lower() == "$and":
                    # Remove the pebblo filter from the $and list
                    for i, _filter in enumerate(value):
                        if pebblo_filter_key in _filter:
                            # This handles Auth filter
                            value.pop(i)
                            break
                        # Check for $not operator with Semantic filter
                        if "$not" in _filter:
                            sem_filter_found = False
                            # This handles Semantic filter
                            for j, nested_filter in enumerate(_filter["$not"]):
                                if pebblo_filter_key in nested_filter:
                                    if len(_filter["$not"]) == 1:
                                        # If only one filter is left,
                                        # then remove the $not operator
                                        value.pop(i)
                                    else:
                                        value[i]["$not"].pop(j)
                                    sem_filter_found = True
                                    break
                            if sem_filter_found:
                                break
                    if len(value) == 1:
                        # If only one filter is left, then remove the $and operator
                        search_kwargs["filter"] = value[0]
                elif key.lower() == "$not":
                    # Remove the pebblo filter from the $not list
                    for i, _filter in enumerate(value):
                        if pebblo_filter_key in _filter:
                            # This removes Semantic filter
                            value.pop(i)
                            break
                    if len(value) == 0:
                        # If no filter is left, then unset the filter
                        search_kwargs["filter"] = {}
                elif key.lower() == "$or":
                    # If $or, pebblo filter will not be present
                    return
            else:
                # Then it's a field, check if it's a pebblo filter
                if key == pebblo_filter_key:
                    filters.pop(key)
                return
        elif len(filters) > 1:
            # Then all keys have to be fields (they cannot be operators)
            if pebblo_filter_key in filters:
                # This handles Auth filter
                filters.pop(pebblo_filter_key)
            return
        else:
            # Got an empty dictionary for filters, ignore the filter
            return
    elif filters is None:
        # If filters is None, ignore the filter
        return
    else:
        raise ValueError(
            f"Invalid filter. Expected a dictionary/None but got type: {type(filters)}"
        )


def _apply_pgvector_semantic_filter(
    search_kwargs: dict, semantic_context: Optional[SemanticContext]
) -> None:
    """
    Set semantic enforcement filter in search_kwargs for PGVector vectorstore.
    """
    # Check if semantic_context is provided
    if semantic_context is not None:
        _semantic_filters = []
        filters = search_kwargs.get("filter")
        if semantic_context.pebblo_semantic_topics is not None:
            # Add pebblo_semantic_topics filter to search_kwargs
            topic_filter: dict = {
                "pebblo_semantic_topics": {
                    "$eq": semantic_context.pebblo_semantic_topics.deny
                }
            }
            _semantic_filters.append(topic_filter)

        if semantic_context.pebblo_semantic_entities is not None:
            # Add pebblo_semantic_entities filter to search_kwargs
            entity_filter: dict = {
                "pebblo_semantic_entities": {
                    "$eq": semantic_context.pebblo_semantic_entities.deny
                }
            }
            _semantic_filters.append(entity_filter)

        if len(_semantic_filters) > 0:
            semantic_filter: dict = {"$not": _semantic_filters}
            _apply_pgvector_filter(search_kwargs, filters, semantic_filter)


def _apply_pgvector_authorization_filter(
    search_kwargs: dict, auth_context: Optional[AuthContext]
) -> None:
    """
    Set identity enforcement filter in search_kwargs for PGVector vectorstore.
    """
    if auth_context is not None:
        auth_filter: dict = {"authorized_identities": {"$eq": auth_context.user_auth}}
        filters = search_kwargs.get("filter")
        _apply_pgvector_filter(search_kwargs, filters, auth_filter)


def _set_identity_enforcement_filter(
    retriever: VectorStoreRetriever, auth_context: Optional[AuthContext]
) -> None:
    """
    Set identity enforcement filter in search_kwargs.

    This method sets the identity enforcement filter in the search_kwargs
    of the retriever based on the type of the vectorstore.
    """
    search_kwargs = retriever.search_kwargs
    if retriever.vectorstore.__class__.__name__ == PINECONE:
        _apply_pinecone_authorization_filter(search_kwargs, auth_context)
    elif retriever.vectorstore.__class__.__name__ == QDRANT:
        _apply_qdrant_authorization_filter(search_kwargs, auth_context)
    elif retriever.vectorstore.__class__.__name__ == PGVECTOR:
        _apply_pgvector_authorization_filter(search_kwargs, auth_context)


def _set_semantic_enforcement_filter(
    retriever: VectorStoreRetriever, semantic_context: Optional[SemanticContext]
) -> None:
    """
    Set semantic enforcement filter in search_kwargs.

    This method sets the semantic enforcement filter in the search_kwargs
    of the retriever based on the type of the vectorstore.
    """
    search_kwargs = retriever.search_kwargs
    if retriever.vectorstore.__class__.__name__ == PINECONE:
        _apply_pinecone_semantic_filter(search_kwargs, semantic_context)
    elif retriever.vectorstore.__class__.__name__ == QDRANT:
        _apply_qdrant_semantic_filter(search_kwargs, semantic_context)
    elif retriever.vectorstore.__class__.__name__ == PGVECTOR:
        _apply_pgvector_semantic_filter(search_kwargs, semantic_context)
