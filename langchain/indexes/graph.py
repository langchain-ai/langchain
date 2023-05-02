"""Graph Index Creator."""
from typing import Optional, Type

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.graphs.networkx_graph import NetworkxEntityGraph, parse_triples
from langchain.indexes.prompts.knowledge_triplet_extraction import (
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)
from langchain.llms.base import BaseLLM


class GraphIndexCreator(BaseModel):
    """Functionality to create graph index."""

    llm: Optional[BaseLLM] = None
    graph_type: Type[NetworkxEntityGraph] = NetworkxEntityGraph

    def from_text(self, text: str) -> NetworkxEntityGraph:
        """Create graph index from text."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        chain = LLMChain(llm=self.llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)
        output = chain.predict(text=text)
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph
