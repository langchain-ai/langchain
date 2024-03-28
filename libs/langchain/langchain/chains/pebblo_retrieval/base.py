"""
Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering
against a vector database.
"""

from typing import List

from langchain_community.vectorstores import Pinecone, Qdrant
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.chains.retrieval_qa.base import RetrievalQA

SUPPORTED_VECTORSTORES = [Pinecone, Qdrant]


class PebbloRetrievalQA(RetrievalQA):
    """
    Retrieval Chain with Identity & Semantic Enforcement for question-answering
    against a vector database.
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStore to use for retrieval."""

    auth_context: dict = Field(exclude=True)
    """Authentication context for identity enforcement."""

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "pebblo_retrieval_qa"

    @validator("auth_context")
    def validate_auth_context(cls, auth_context: dict) -> dict:
        """
        Validate auth_context
        """
        if "authorized_identities" not in auth_context:
            raise ValueError("auth_context must contain 'authorized_identities'")
        if not isinstance(auth_context["authorized_identities"], list):
            raise ValueError("authorized_identities must be a list")
        return auth_context

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
        #  Set identity enforcement filter in search_kwargs
        self.set_identity_enforcement_filter()
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

    def set_identity_enforcement_filter(self) -> None:
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
        else:
            raise ValueError(
                f"Vectorstore must be an instance of one of the supported "
                f"vectorstores: {SUPPORTED_VECTORSTORES}. "
                f"Got {type(self.retriever.vectorstore).__name__} instead."
            )

    def _apply_pinecone_authorization_filter(self, search_kwargs: dict) -> None:
        """
        Set identity enforcement filter in search_kwargs for Pinecone vectorstore.
        """
        if (
            "filter" in search_kwargs
            and "authorized_identities" in search_kwargs["filter"]
        ):
            raise ValueError(
                "authorized_identities already exists in search_kwargs['filter']"
            )
        search_kwargs.setdefault("filter", {})["authorized_identities"] = {
            "$in": self.auth_context["authorized_identities"]
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
        # Create a identity enforcement filter condition
        identity_enforcement_filter = rest.FieldCondition(
            key="metadata.authorized_identities",
            match=rest.MatchAny(any=self.auth_context["authorized_identities"]),
        )
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
                            raise ValueError(
                                "Filter for 'authorized_identities' already exists"
                            )
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
