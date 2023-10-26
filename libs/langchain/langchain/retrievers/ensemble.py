"""
Ensemble retriever that ensemble the results of 
multiple retrievers by using weighted  Reciprocal Rank Fusion
"""
from typing import Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.pydantic_v1 import root_validator
from langchain.retrievers.rrf import weighted_reciprocal_rank_fusion
from langchain.schema import BaseRetriever, Document


class EnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
    """

    retrievers: List[BaseRetriever]
    weights: List[float]
    c: int = 60

    @root_validator(pre=True)
    def set_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("weights"):
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        return values

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = self.rank_fusion(query, run_manager)

        return fused_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = await self.arank_fusion(query, run_manager)

        return fused_documents

    def rank_fusion(
        self, query: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents

    async def arank_fusion(
        self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            await retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents

    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        return weighted_reciprocal_rank_fusion(
            doc_lists, weights=self.weights, c=self.c
        )
