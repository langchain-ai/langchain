"""Question answering over a graph."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import ENTITY_EXTRACTION_PROMPT, PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.networkx_graph import NetworkxEntityGraph, get_entities
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


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
        cls,
        llm: BaseLLM,
        qa_prompt: BasePromptTemplate = PROMPT,
        entity_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT,
        **kwargs: Any,
    ) -> GraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

        return cls(
            qa_chain=qa_chain,
            entity_extraction_chain=entity_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Extract entities, look up info and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        entity_string = self.entity_extraction_chain.run(question)

        _run_manager.on_text("Entities Extracted:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            entity_string, color="green", end="\n", verbose=self.verbose
        )
        entities = get_entities(entity_string)
        context = ""
        for entity in entities:
            triplets = self.graph.get_entity_knowledge(entity)
            context += "\n".join(triplets)
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: result[self.qa_chain.output_key]}
