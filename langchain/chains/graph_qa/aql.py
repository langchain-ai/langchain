"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import AQL_GENERATION_PROMPT, AQL_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.arangodb_graph import ArangoDBGraph
from langchain.schema import BasePromptTemplate


class ArangoDBGraphQAChain(Chain):
    """Chain for question-answering against a graph by generating AQL statements."""

    graph: ArangoDBGraph = Field(exclude=True)
    aql_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    # Class Variable specifying the number of AQL Query Results to return
    top_k = 10

    return_aql_result: bool = False

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "graph_aql_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        qa_prompt: BasePromptTemplate = AQL_QA_PROMPT,
        aql_prompt: BasePromptTemplate = AQL_GENERATION_PROMPT,
        **kwargs: Any,
    ) -> ArangoDBGraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        aql_generation_chain = LLMChain(llm=llm, prompt=aql_prompt)

        return cls(
            qa_chain=qa_chain,
            aql_generation_chain=aql_generation_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Generate an AQL statement from user input, use it retrieve a response
        from an ArangoDB Database instance, and respond to the user input
        in natural language.
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        user_input = inputs[self.input_key]

        aql_generation_response = self.aql_generation_chain.run(
            {"user_input": user_input, "adb_schema": self.graph.schema},
            callbacks=callbacks,
        )

        # breakpoint()
        # matches = aql_generation_response.split('|')
        # if not matches or (len(matches) == 1 and "RETURN" not in matches[0]):

        pattern = r"```(?i)aql(.*?)```"
        matches = re.findall(pattern, aql_generation_response, re.DOTALL)
        if not matches:
            # Cannot parse AQL Query (if any)
            return {self.output_key: aql_generation_response}

        aql_query = matches[0]
        # results = []
        # for aql_query in matches
        # results.append(result[self.qa_chain.output_key])
        # return {self.output_key: " ".join(results)}

        # breakpoint()
        _run_manager.on_text("Generated AQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(aql_query, color="green", end="\n", verbose=self.verbose)

        # Retrieve and limit the number of results
        # breakpoint()
        aql_result = self.graph.query(aql_query, self.top_k)
        # breakpoint()

        _run_manager.on_text("AQL Result:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(aql_result), color="green", end="\n", verbose=self.verbose
        )

        result = self.qa_chain(
            {
                "user_input": user_input,
                "aql_query": aql_query,
                "aql_result": aql_result,
            },
            callbacks=callbacks,
        )
        # breakpoint()

        result = {self.output_key: result[self.qa_chain.output_key]}
        if self.return_aql_result:
            result["aql_result"] = aql_result

        return result
