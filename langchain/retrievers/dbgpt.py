from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class DBGPTRetriever(BaseRetriever):
    """Retriever that uses the DB-GPT embedding engine API.
    embedding_model_path: embedding model path eg:'your_model_path/all-MiniLM-L6-v2'
    vector_store_config: vector database config, it contains:
    vector_store_type:str : Chroma or Milvus
    Chroma eg:
    vector_store_config = {
        "vector_store_type":"Chroma",
        "vector_store_name":"dbgpt",
        "chroma_persist_path":"your_path"
    }
    Milvus eg:
    vector_store_config = {
        "vector_store_type":"Milvus",
        "milvus_url":"your_url",
        "milvus_port":"your_port",
        "milvus_username":"your_username",(optional)
        "milvus_password":"your_password",(optional)
    }
    """

    embedding_model_path: str
    vector_store_config: {}
    query_kwargs: Dict = Field(default_factory=dict)

    def __init__(
        self,
        embedding_model_path: str,
        vector_store_config: {},
        top_k: Optional[int] = None,
    ):
        self.vector_store_config = vector_store_config
        self.embedding_model_path = embedding_model_path
        self.top_k = top_k

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from pilot import EmbeddingEngine
        except ImportError:
            raise ImportError(
                "You need to install `pip install db-gpt` to use this retriever."
            )
        embedding_engine = EmbeddingEngine(
            model_name=self.embedding_model_path,
            vector_store_config=self.vector_store_config,
        )
        docs = embedding_engine.similar_search(query, self.top_k)
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("DBGPTRetriever does not support async")
