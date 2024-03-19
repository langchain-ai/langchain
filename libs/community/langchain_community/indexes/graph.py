"""Graph Index Creator."""
from typing import Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from langchain_community.graphs.networkx_graph import NetworkxEntityGraph, parse_triples
from langchain_community.indexes._prompts.knowledge_triplet_extraction import (
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)


class GraphIndexCreator(BaseModel):
    """Functionality to create graph index."""

    llm: Optional[BaseLanguageModel] = None
    graph_type: Type[NetworkxEntityGraph] = NetworkxEntityGraph

    def from_text(
        self, text: str, prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    ) -> NetworkxEntityGraph:
        """Create graph index from text."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        chain = prompt | self.llm | StrOutputParser()
        output = chain.invoke({"text": text})
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph

    async def afrom_text(
        self, text: str, prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    ) -> NetworkxEntityGraph:
        """Create graph index from text asynchronously."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        chain = prompt | self.llm | StrOutputParser()
        output = await chain.ainvoke({"text": text})
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph
