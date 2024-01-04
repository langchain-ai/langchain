from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document


# TODO: this is a work in progress
# TODO: do we need `metadata` and `tags` in interface?
# TODO: do we need other methods from BaseRetriever?
class RetrieverInterface(ABC):
    """Interface for a Document retrieval system.

    The default implementation of this interface is :class:`BaseRetriever`.
    A retrieval system is defined as something that can take string queries and return
        the most 'relevant' Documents from some source.

    Example:
        .. code-block:: python

            class TFIDFRetriever(BaseRetriever, BaseModel):
                vectorizer: Any
                docs: List[Document]
                tfidf_array: Any
                k: int = 4

                class Config:
                    arbitrary_types_allowed = True

                def get_relevant_documents(self, query: str) -> List[Document]:
                    from sklearn.metrics.pairwise import cosine_similarity

                    # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                    query_vec = self.vectorizer.transform([query])
                    # Op -- (n_docs,1) -- Cosine Sim with each doc
                    results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                    return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
    """  # noqa: E501

    tags: Optional[List[str]] = None
    """Optional list of tags associated with the retriever. Defaults to None
    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the retriever. Defaults to None
    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """
