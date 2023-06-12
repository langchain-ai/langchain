"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import RDF_SPARQL_GENERATION_PROMPT, RDF_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.rdf_graph import RDFGraph
from langchain.prompts.base import BasePromptTemplate

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

class GraphRDFQAChain(Chain):
    """Chain for question-answering against a graph by generating RDF SPARQL statements."""

    graph: RDFGraph = Field(exclude=True)
    rdf_sparql_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""

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
        llm: BaseLanguageModel,
        *,
        qa_prompt: BasePromptTemplate = RDF_QA_PROMPT,
        rdf_sparql_prompt: BasePromptTemplate = RDF_SPARQL_GENERATION_PROMPT,
        **kwargs: Any,
    ) -> GraphRDFQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        rdf_sparql_generation_chain = LLMChain(llm=llm, prompt=rdf_sparql_prompt)

        return cls(
            qa_chain=qa_chain,
            rdf_sparql_generation_chain=rdf_sparql_generation_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate RDF SPARQL statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        intermediate_steps: List = []

        generated_rdf_sparql = self.rdf_sparql_generation_chain.run(
            {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        _run_manager.on_text("Generated RDF SPARQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_rdf_sparql, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_rdf_sparql})

        # Retrieve and limit the number of results
        context = self.graph.query(generated_rdf_sparql)[: self.top_k]

        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})

            result = self.qa_chain(
                {"question": question, "context": context},
                callbacks=callbacks,
            )
            final_result = result[self.qa_chain.output_key]

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
