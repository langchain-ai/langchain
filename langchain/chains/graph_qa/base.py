from __future__ import annotations
from langchain.graphs import NetworkxEntityGraph

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from typing import List, Dict, Any
from pydantic import Field
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.graph_qa.prompts import ENTITY_EXTRACTION_PROMPT, PROMPT

class GraphQAChain(Chain):
    """Chain for question-answering against a graph."""

    graph: NetworkxEntityGraph = Field(exclude=True)
    entity_extraction_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, qa_prompt: BasePromptTemplate = PROMPT, entity_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT, **kwargs: Any
    ) -> GraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

        return cls(qa_chain=qa_chain, entity_chain=entity_chain, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]
        entities = self.entity_extraction_chain.run(question)
        context = ""
        for entity in entities:
            context += self.graph.get_entity_knowledge(entity)
        result = self.qa_chain({"question": question, "context": context})
        return {self.output_key: result[self.qa_chain.output_key]}
