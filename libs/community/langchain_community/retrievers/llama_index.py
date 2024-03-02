from typing import Any, Dict, List, cast

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever


class LlamaIndexRetriever(BaseRetriever):
    """`LlamaIndex` retriever.

    It is used for the question-answering with sources over
    an LlamaIndex data structure."""

    index: Any
    """LlamaIndex index to query."""
    query_kwargs: Dict = Field(default_factory=dict)
    """Keyword arguments to pass to the query method."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.core.base.response.schema import Response
            from llama_index.core.indices.base import BaseGPTIndex
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseGPTIndex, self.index)

        response = index.query(query, **self.query_kwargs)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.get_content(), metadata=metadata)
            )
        return docs


class LlamaIndexGraphRetriever(BaseRetriever):
    """`LlamaIndex` graph data structure retriever.

    It is used for question-answering with sources over an LlamaIndex
    graph data structure."""

    graph: Any
    """LlamaIndex graph to query."""
    query_configs: List[Dict] = Field(default_factory=list)
    """List of query configs to pass to the query method."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.core.base.response.schema import Response
            from llama_index.core.composability.base import (
                QUERY_CONFIG_TYPE,
                ComposableGraph,
            )
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        graph = cast(ComposableGraph, self.graph)

        # for now, inject response_mode="no_text" into query configs
        for query_config in self.query_configs:
            query_config["response_mode"] = "no_text"
        query_configs = cast(List[QUERY_CONFIG_TYPE], self.query_configs)
        response = graph.query(query, query_configs=query_configs)
        response = cast(Response, response)

        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.get_content(), metadata=metadata)
            )
        return docs
