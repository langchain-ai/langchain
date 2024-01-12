"""
Question answering over a GraphDB graph using SPARQL.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import rdflib
from langchain_community.graphs import GraphDBGraph
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from pyparsing import ParseException

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
    GRAPHDB_FIX_SELECT_PROMPT,
    GRAPHDB_GENERATION_SELECT_PROMPT,
    GRAPHDB_QA_PROMPT,
)
from langchain.chains.llm import LLMChain


class GraphDBQAChain(Chain):
    """Question-answering against a GraphDB graph by generating SPARQL statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: GraphDBGraph = Field(exclude=True)
    sparql_generation_select_chain: LLMChain
    sparql_fix_select_chain: LLMChain
    qa_chain: LLMChain
    max_regeneration_attempts: int
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        _output_keys = [self.output_key]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        sparql_select_prompt: BasePromptTemplate = GRAPHDB_GENERATION_SELECT_PROMPT,
        sparql_fix_select_prompt: BasePromptTemplate = GRAPHDB_FIX_SELECT_PROMPT,
        qa_prompt: BasePromptTemplate = GRAPHDB_QA_PROMPT,
        max_regeneration_attempts: int = 5,
        **kwargs: Any,
    ) -> GraphDBQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        sparql_generation_select_chain = LLMChain(llm=llm, prompt=sparql_select_prompt)
        sparql_fix_select_chain = LLMChain(llm=llm, prompt=sparql_fix_select_prompt)
        max_regeneration_attempts = max_regeneration_attempts
        return cls(
            qa_chain=qa_chain,
            sparql_generation_select_chain=sparql_generation_select_chain,
            sparql_fix_select_chain=sparql_fix_select_chain,
            max_regeneration_attempts=max_regeneration_attempts,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """
        Generate a SPARQL query, use it to retrieve a response from GraphDB and answer
        the question.
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        prompt = inputs[self.input_key]

        generated_sparql = self.sparql_generation_select_chain.run(
            {"prompt": prompt, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        generated_sparql = self._get_valid_sparql_query(
            _run_manager, callbacks, generated_sparql
        )
        query_results = self._execute_sparql_query(_run_manager, generated_sparql)

        result = self.qa_chain(
            {"prompt": prompt, "context": query_results}, callbacks=callbacks
        )
        res = result[self.qa_chain.output_key]
        return {self.output_key: res}

    def _get_valid_sparql_query(
        self,
        _run_manager: CallbackManagerForChainRun,
        callbacks: CallbackManager,
        generated_sparql: str,
    ) -> str:
        try:
            return self._parse_sparql_query(_run_manager, generated_sparql)
        except ParseException as e:
            retries = 0
            parse_exception = str(e)
            self._log_invalid_sparql_query(
                _run_manager, generated_sparql, parse_exception
            )

            while retries < self.max_regeneration_attempts:
                try:
                    generated_sparql = self.sparql_fix_select_chain.run(
                        {
                            "parse_exception": parse_exception,
                            "generated_sparql": generated_sparql,
                        },
                        callbacks=callbacks,
                    )
                    return self._parse_sparql_query(_run_manager, generated_sparql)
                except ParseException as e:
                    retries += 1
                    parse_exception = str(e)
                    self._log_invalid_sparql_query(
                        _run_manager, generated_sparql, parse_exception
                    )

        raise ValueError("The generated SPARQL query is invalid.")

    def _parse_sparql_query(
        self, _run_manager: CallbackManagerForChainRun, generated_sparql: str
    ) -> str:
        from rdflib.plugins.sparql import prepareQuery

        prepareQuery(generated_sparql)
        self._log_valid_sparql_query(_run_manager, generated_sparql)
        return generated_sparql

    def _log_valid_sparql_query(
        self, _run_manager: CallbackManagerForChainRun, generated_query: str
    ) -> None:
        _run_manager.on_text("Generated SPARQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_query, color="green", end="\n", verbose=self.verbose
        )

    def _log_invalid_sparql_query(
        self,
        _run_manager: CallbackManagerForChainRun,
        generated_query: str,
        error_message: str,
    ) -> None:
        _run_manager.on_text("Invalid SPARQL query: ", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_query, color="red", end="\n", verbose=self.verbose
        )
        _run_manager.on_text(
            "SPARQL Query Parse Error: ", end="\n", verbose=self.verbose
        )
        _run_manager.on_text(
            error_message, color="red", end="\n\n", verbose=self.verbose
        )

    def _execute_sparql_query(
        self, _run_manager: CallbackManagerForChainRun, generated_query: str
    ) -> List[rdflib.query.ResultRow]:
        query_results = self.graph.query(generated_query)
        _run_manager.on_text("Query results:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(query_results), color="green", end="\n", verbose=self.verbose
        )
        return query_results
