import warnings
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores.zilliz import Zilliz

# TODO: Update to ZillizClient + Hybrid Search when available


class ZillizRetriever(BaseRetriever):
    """`Zilliz API` retriever."""

    embedding_function: Embeddings
    """The underlying embedding function from which documents will be retrieved."""
    collection_name: str = "LangChainCollection"
    """The name of the collection in Zilliz."""
    connection_args: Optional[Dict[str, Any]] = None
    """The connection arguments for the Zilliz client."""
    consistency_level: str = "Session"
    """The consistency level for the Zilliz client."""
    search_params: Optional[dict] = None
    """The search parameters for the Zilliz client."""
    store: Zilliz
    """The underlying Zilliz store."""
    retriever: BaseRetriever
    """The underlying retriever."""

    @root_validator(pre=True)
    def create_client(cls, values: dict) -> dict:
        values["store"] = Zilliz(
            values["embedding_function"],
            values["collection_name"],
            values["connection_args"],
            values["consistency_level"],
        )
        values["retriever"] = values["store"].as_retriever(
            search_kwargs={"param": values["search_params"]}
        )
        return values

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Add text to the Zilliz store

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.store.add_texts(texts, metadatas)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        return self.retriever.get_relevant_documents(
            query, run_manager=run_manager.get_child(), **kwargs
        )


def ZillizRetreiver(*args: Any, **kwargs: Any) -> ZillizRetriever:
    """Deprecated ZillizRetreiver.

    Please use ZillizRetriever ('i' before 'e') instead.

    Args:
        *args:
        **kwargs:

    Returns:
        ZillizRetriever
    """
    warnings.warn(
        "ZillizRetreiver will be deprecated in the future. "
        "Please use ZillizRetriever ('i' before 'e') instead.",
        DeprecationWarning,
    )
    return ZillizRetriever(*args, **kwargs)
