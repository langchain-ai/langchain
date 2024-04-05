"""
Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering
against a vector database.
"""

import logging
from typing import Any, List, Optional, Union

from langchain_community.vectorstores import Pinecone, Qdrant
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.chains.pebblo_retrieval.models import AuthContext, SemanticContext
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA

logger = logging.getLogger(__name__)

SUPPORTED_VECTORSTORES = [Pinecone, Qdrant]

AUTH_IDENTITY_OVERRIDE_MSG = (
    "'authorized_identities' filter in 'search_kwargs' will be overridden when using "
    "PebbloRetrievalQA chain with 'auth_context'. Please avoid adding it in "
    "'search_kwargs'."
)

SEMANTIC_TOPIC_OVERRIDE_MSG = (
    "'pebblo_semantic_topics' filter in 'search_kwargs' will be overridden when using "
    "PebbloRetrievalQA chain with 'semantic_context'. Please avoid adding it in "
    "'search_kwargs'."
)

SEMANTIC_ENTITY_OVERRIDE_MSG = (
    "'pebblo_semantic_entities' filter in 'search_kwargs' will be overridden when "
    "using PebbloRetrievalQA chain with 'semantic_context'. Please avoid adding it in "
    "'search_kwargs'."
)


class PebbloRetrievalQA(RetrievalQA):
    """
    Retrieval Chain with Identity & Semantic Enforcement for question-answering
    against a vector database.
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStore to use for retrieval."""

    auth_context: Optional[AuthContext] = Field(None, exclude=True)
    """Authentication context for identity enforcement."""

    semantic_context: Optional[SemanticContext] = Field(None, exclude=True)
    """Semantic context for semantic enforcement."""

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "pebblo_retrieval_qa"

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> "PebbloRetrievalQA":
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm, chain_type=chain_type, **_chain_type_kwargs
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @validator("retriever", pre=True, always=True)
    def validate_vectorstore(
        cls, retriever: VectorStoreRetriever
    ) -> VectorStoreRetriever:
        """
        Validate that the vectorstore of the retriever is supported vectorstores.
        """
        if not any(
            isinstance(retriever.vectorstore, supported_class)
            for supported_class in SUPPORTED_VECTORSTORES
        ):
            raise ValueError(
                f"Vectorstore must be an instance of one of the supported "
                f"vectorstores: {SUPPORTED_VECTORSTORES}. "
                f"Got {type(retriever.vectorstore).__name__} instead."
            )
        return retriever

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        if self.auth_context is not None:
            self._set_identity_enforcement_filter()
        if self.semantic_context is not None:
            self._set_semantic_enforcement_filter()
        docs = super()._get_docs(question, run_manager=run_manager)
        return docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        raise NotImplementedError("PebbloRetrievalQA does not support async")

    def _set_identity_enforcement_filter(self) -> None:
        """
        Set identity enforcement filter in search_kwargs.

        This method sets the identity enforcement filter in the search_kwargs
        of the retriever based on the type of the vectorstore.
        """
        search_kwargs = self.retriever.search_kwargs
        if isinstance(self.retriever.vectorstore, Pinecone):
            self._apply_pinecone_authorization_filter(search_kwargs)
        elif isinstance(self.retriever.vectorstore, Qdrant):
            self._apply_qdrant_authorization_filter(search_kwargs)

    def _set_semantic_enforcement_filter(self) -> None:
        """
        Set semantic enforcement filter in search_kwargs.

        This method sets the semantic enforcement filter in the search_kwargs
        of the retriever based on the type of the vectorstore.
        """
        search_kwargs = self.retriever.search_kwargs
        if isinstance(self.retriever.vectorstore, Pinecone):
            self._apply_pinecone_semantic_filter(search_kwargs)
        elif isinstance(self.retriever.vectorstore, Qdrant):
            self._apply_qdrant_semantic_filter(search_kwargs)

    def _apply_pinecone_authorization_filter(self, search_kwargs: dict) -> None:
        """
        Set identity enforcement filter in search_kwargs for Pinecone vectorstore.
        """
        if search_kwargs.get("filter", {}).get("authorized_identities"):
            logger.warning(AUTH_IDENTITY_OVERRIDE_MSG)

        if self.auth_context is not None:
            search_kwargs.setdefault("filter", {})["authorized_identities"] = {
                "$in": self.auth_context.authorized_identities
            }

    def _apply_pinecone_semantic_filter(self, search_kwargs: dict) -> None:
        """
        Set semantic enforcement filter in search_kwargs for Pinecone vectorstore.
        """
        # Check if semantic_context is provided
        semantic_context = self.semantic_context
        if semantic_context is not None:
            if semantic_context.pebblo_semantic_topics is not None:
                # Check if pebblo_semantic_topics filter is overridden
                if search_kwargs.get("filter", {}).get("pebblo_semantic_topics"):
                    logger.warning(SEMANTIC_TOPIC_OVERRIDE_MSG)

                # Add pebblo_semantic_topics filter to search_kwargs
                search_kwargs.setdefault("filter", {})["pebblo_semantic_topics"] = {
                    "$nin": semantic_context.pebblo_semantic_topics.deny
                }

            if semantic_context.pebblo_semantic_entities is not None:
                # Check if pebblo_semantic_entities filter is overridden
                if search_kwargs.get("filter", {}).get("pebblo_semantic_entities"):
                    logger.warning(SEMANTIC_ENTITY_OVERRIDE_MSG)

                # Add pebblo_semantic_entities filter to search_kwargs
                search_kwargs.setdefault("filter", {})["pebblo_semantic_entities"] = {
                    "$nin": semantic_context.pebblo_semantic_entities.deny
                }

    def _apply_qdrant_authorization_filter(self, search_kwargs: dict) -> None:
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

        if self.auth_context is not None:
            # Create a identity enforcement filter condition
            identity_enforcement_filter = rest.FieldCondition(
                key="metadata.authorized_identities",
                match=rest.MatchAny(any=self.auth_context.authorized_identities),
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
                    # Error if 'authorized_identities' filter is already present
                    for condition in existing_filter.must:
                        if (
                            hasattr(condition, "key")
                            and condition.key == "metadata.authorized_identities"
                        ):
                            logger.warning(AUTH_IDENTITY_OVERRIDE_MSG)

                    # Add identity enforcement filter to 'must' conditions
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

    def _apply_qdrant_semantic_filter(self, search_kwargs: dict) -> None:
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
            self.semantic_context is not None
            and self.semantic_context.pebblo_semantic_topics is not None
        ):
            semantic_topics_filter = rest.FieldCondition(
                key="metadata.pebblo_semantic_topics",
                match=rest.MatchAny(
                    any=self.semantic_context.pebblo_semantic_topics.deny
                ),
            )
            semantic_filters.append(semantic_topics_filter)
        if (
            self.semantic_context is not None
            and self.semantic_context.pebblo_semantic_entities is not None
        ):
            semantic_entities_filter = rest.FieldCondition(
                key="metadata.pebblo_semantic_entities",
                match=rest.MatchAny(
                    any=self.semantic_context.pebblo_semantic_entities.deny
                ),
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
                    for condition in existing_filter.must_not:
                        if hasattr(condition, "key"):
                            if condition.key == "metadata.pebblo_semantic_topics":
                                logger.warning(SEMANTIC_TOPIC_OVERRIDE_MSG)
                            if condition.key == "metadata.pebblo_semantic_entities":
                                logger.warning(SEMANTIC_ENTITY_OVERRIDE_MSG)
                    # Add semantic enforcement filters to 'must_not' conditions
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
