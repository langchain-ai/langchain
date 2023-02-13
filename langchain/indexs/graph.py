
from typing import Optional, Type
from pydantic import BaseModel
from langchain.graphs.networkx_graph import NetworkxEntityGraph, _parse_triples
from langchain.llms.base import BaseLLM
from langchain.chains.llm import LLMChain
from langchain.indexs.prompts.knowledge_triplet_extraction import KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT

class GraphIndexCreator(BaseModel):

    llm: Optional[BaseLLM] = None
    graph_type: Type[NetworkxEntityGraph] = NetworkxEntityGraph

    def from_text(self, text: str) -> NetworkxEntityGraph:
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        chain = LLMChain(llm=self.llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)
        output = chain.predict(text=text)
        knowledge = _parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph


