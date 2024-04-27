"""Question answering over a graph."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import rdflib

from langchain_community.graphs import OntotextGraphDBGraph
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
    GRAPHDB_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_PROMPT,
    GRAPHDB_SPARQL_GENERATION_PROMPT,
)
from langchain.chains.llm import LLMChain


class OntotextGraphDBQAChain(Chain):
    """Question-answering against Ontotext GraphDB
       https://graphdb.ontotext.com/ by generating SPARQL queries.

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

    graph: OntotextGraphDBGraph = Field(exclude=True)
    sparql_generation_chain: LLMChain
    sparql_fix_chain: LLMChain
    max_fix_retries: int
    qa_chain: LLMChain
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
        sparql_generation_prompt: BasePromptTemplate = GRAPHDB_SPARQL_GENERATION_PROMPT,
        sparql_fix_prompt: BasePromptTemplate = GRAPHDB_SPARQL_FIX_PROMPT,
        max_fix_retries: int = 5,
        qa_prompt: BasePromptTemplate = GRAPHDB_QA_PROMPT,
        **kwargs: Any,
    ) -> OntotextGraphDBQAChain:
        """Initialize from LLM."""
        sparql_generation_chain = LLMChain(llm=llm, prompt=sparql_generation_prompt)
        sparql_fix_chain = LLMChain(llm=llm, prompt=sparql_fix_prompt)
        max_fix_retries = max_fix_retries
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        return cls(
            qa_chain=qa_chain,
            sparql_generation_chain=sparql_generation_chain,
            sparql_fix_chain=sparql_fix_chain,
            max_fix_retries=max_fix_retries,
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
        ontology_schema = self.graph.get_schema

        sparql_generation_chain_result = self.sparql_generation_chain.invoke(
            {"prompt": prompt, "schema": ontology_schema}, callbacks=callbacks
        )
        generated_sparql = sparql_generation_chain_result[
            self.sparql_generation_chain.output_key
        ]

        generated_sparql = self._get_prepared_sparql_query(
            _run_manager, callbacks, generated_sparql, ontology_schema
        )
        query_results = self._execute_query(generated_sparql)

        qa_chain_result = self.qa_chain.invoke(
            {"prompt": prompt, "context": query_results}, callbacks=callbacks
        )
        result = qa_chain_result[self.qa_chain.output_key]
        return {self.output_key: result}

    def _get_prepared_sparql_query(
        self,
        _run_manager: CallbackManagerForChainRun,
        callbacks: CallbackManager,
        generated_sparql: str,
        ontology_schema: str,
    ) -> str:
        try:
            return self._prepare_sparql_query(_run_manager, generated_sparql)
        except Exception as e:
            retries = 0
            error_message = str(e)
            self._log_invalid_sparql_query(
                _run_manager, generated_sparql, error_message
            )

            while retries < self.max_fix_retries:
                try:
                    sparql_fix_chain_result = self.sparql_fix_chain.invoke(
                        {
                            "error_message": error_message,
                            "generated_sparql": generated_sparql,
                            "schema": ontology_schema,
                        },
                        callbacks=callbacks,
                    )
                    generated_sparql = sparql_fix_chain_result[
                        self.sparql_fix_chain.output_key
                    ]
                    return self._prepare_sparql_query(_run_manager, generated_sparql)
                except Exception as e:
                    retries += 1
                    parse_exception = str(e)
                    self._log_invalid_sparql_query(
                        _run_manager, generated_sparql, parse_exception
                    )

        raise ValueError("The generated SPARQL query is invalid.")

    def _prepare_sparql_query(
        self, _run_manager: CallbackManagerForChainRun, generated_sparql: str
    ) -> str:
        from rdflib.plugins.sparql import prepareQuery

        prepareQuery(generated_sparql)
        self._log_prepared_sparql_query(_run_manager, generated_sparql)
        return generated_sparql

    def _log_prepared_sparql_query(
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

    def _execute_query(self, query: str) -> List[rdflib.query.ResultRow]:
        try:
            return self.graph.query(query)
        except Exception:
            raise ValueError("Failed to execute the generated SPARQL query.")
