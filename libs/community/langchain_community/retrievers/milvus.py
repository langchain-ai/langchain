"""Milvus Retriever"""
import warnings
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

from langchain_community.vectorstores.milvus import Milvus

# TODO: Update to MilvusClient + Hybrid Search when available


class MilvusRetriever(BaseRetriever):
    """`Milvus API` retriever."""

    embedding_function: Embeddings
    collection_name: str = "LangChainCollection"
    connection_args: Optional[Dict[str, Any]] = None
    consistency_level: str = "Session"
    search_params: Optional[dict] = None

    store: Milvus
    retriever: BaseRetriever

    @root_validator(pre=True)
    def create_retriever(cls, values: Dict) -> Dict:
        """Create the Milvus store and retriever."""
        values["store"] = Milvus(
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
        """Add text to the Milvus store

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


def MilvusRetreiver(*args: Any, **kwargs: Any) -> MilvusRetriever:
    """Deprecated MilvusRetreiver. Please use MilvusRetriever ('i' before 'e') instead.

    Args:
        *args:
        **kwargs:

    Returns:
        MilvusRetriever
    """
    warnings.warn(
        "MilvusRetreiver will be deprecated in the future. "
        "Please use MilvusRetriever ('i' before 'e') instead.",
        DeprecationWarning,
    )
    return MilvusRetriever(*args, **kwargs)
