from typing import Any, Dict, List, cast

from pydantic import Field

from langchain.schema import BaseRetriever, Document


class LlamaIndexRetriever(BaseRetriever):
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

        response = index.as_query_engine(response_mode="no_text", **self.query_kwargs).query(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata or {}
            docs.append(
                Document(page_content=source_node.node.text, metadata=metadata)
            )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query using async."""
        try:
            from llama_index.indices.base import BaseGPTIndex
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseGPTIndex, self.index)

        response = await index.as_query_engine(response_mode="no_text", **self.query_kwargs).aquery(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata or {}
            docs.append(
                Document(page_content=source_node.node.text, metadata=metadata)
            )
        return docs


class LlamaIndexGraphRetriever(BaseRetriever):
    """Question-answering with sources over an LlamaIndex graph data structure."""

    graph: Any
    query_kwargs: Dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.composability import ComposableGraph
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        graph = cast(ComposableGraph, self.graph)

        custom_query_engines = self.query_kwargs
        custom_query_engines['response_mode'] = "no_text"
        response = graph.as_query_engine(custom_query_engines=custom_query_engines).query(query)
        response = cast(Response, response)

        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata or {}
            docs.append(
                Document(page_content=source_node.node.text, metadata=metadata)
            )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query using async."""
        try:
            from llama_index.composability import ComposableGraph
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        graph = cast(ComposableGraph, self.graph)

        custom_query_engines = self.query_kwargs
        custom_query_engines['response_mode'] = "no_text"
        response = await graph.as_query_engine(custom_query_engines=custom_query_engines).aquery(query)
        response = cast(Response, response)

        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata or {}
            docs.append(
                Document(page_content=source_node.node.text, metadata=metadata)
            )
        return docs
