from typing import Any, Dict, List, cast

from pydantic import BaseModel, Field

from langchain.schema import BaseRetriever, Document


class LlamaIndexRetriever(BaseRetriever, BaseModel):
    """Question-answering with sources over an LlamaIndex data structure."""

    index: Any
    query_kwargs: Dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.indices.base import BaseGPTIndex
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseGPTIndex, self.index)

        response = index.query(query, response_mode="no_text", **self.query_kwargs)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.extra_info or {}
            docs.append(
                Document(page_content=source_node.source_text, metadata=metadata)
            )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("LlamaIndexRetriever does not support async")


class LlamaIndexGraphRetriever(BaseRetriever, BaseModel):
    """Question-answering with sources over an LlamaIndex graph data structure."""

    graph: Any
    query_configs: List[Dict] = Field(default_factory=list)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.composability.graph import (
                QUERY_CONFIG_TYPE,
                ComposableGraph,
            )
            from llama_index.response.schema import Response
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
            metadata = source_node.extra_info or {}
            docs.append(
                Document(page_content=source_node.source_text, metadata=metadata)
            )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("LlamaIndexGraphRetriever does not support async")
