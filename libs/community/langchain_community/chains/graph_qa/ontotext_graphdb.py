"""Question answering over a graph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import SPARQLWrapper
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field

from langchain_community.chains.graph_qa.prompts import (
    GRAPHDB_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_PROMPT,
    GRAPHDB_SPARQL_GENERATION_PROMPT,
)
from langchain_community.graphs import OntotextGraphDBGraph


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

    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests.

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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

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

        query_results = self.__execute_query(
            generated_sparql, ontology_schema, _run_manager, callbacks
        )
        query_results_str = self.query_results_to_string(query_results, _run_manager)

        qa_chain_result = self.qa_chain.invoke(
            {"prompt": prompt, "context": query_results_str}, callbacks=callbacks
        )
        result = qa_chain_result[self.qa_chain.output_key]
        return {self.output_key: result}

    def __execute_query(
        self,
        generated_sparql: str,
        ontology_schema: str,
        _run_manager: CallbackManagerForChainRun,
        callbacks: CallbackManager,
    ) -> Union[str, SPARQLWrapper.SmartWrapper.Bindings]:
        """
        Executes the generated SPARQL query.
        In case of invalid SPARQL query in terms of syntax,
        try to generate the query again up to a certain number of times.
        """

        from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

        try:
            _run_manager.on_text(
                "Generated query:",
                end="\n",
                verbose=self.verbose,
            )
            _run_manager.on_text(
                generated_sparql,
                end="\n",
                verbose=self.verbose,
            )
            return self.graph.query(generated_sparql)
        except QueryBadFormed as e:  # status code 400
            retries = 1
            error_message = str(e)
            _run_manager.on_text(
                f"QueryBadFormed (status code 400): {error_message}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )

            if retries > self.max_fix_retries:
                raise e
            while retries <= self.max_fix_retries:
                _run_manager.on_text(
                    f"Retrying to generate the query {retries}/{self.max_fix_retries}",
                    color="red",
                    end="\n",
                    verbose=self.verbose,
                )
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
                    _run_manager.on_text(
                        "Generated query:",
                        end="\n",
                        verbose=self.verbose,
                    )
                    _run_manager.on_text(
                        generated_sparql,
                        end="\n",
                        verbose=self.verbose,
                    )
                    return self.graph.query(generated_sparql)
                except QueryBadFormed as e:
                    retries += 1
                    error_message = str(e)
                    _run_manager.on_text(
                        f"QueryBadFormed (status code 400): {error_message}",
                        color="red",
                        end="\n",
                        verbose=self.verbose,
                    )
                    if retries > self.max_fix_retries:
                        raise e

        except Exception as e:
            _run_manager.on_text(
                f"Exception: {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
            raise e

        _run_manager.on_text(
            "Unable to execute query",
            color="red",
            end="\n",
            verbose=self.verbose,
        )
        raise Exception

    def query_results_to_string(
        self,
        query_results: Union[str, SPARQLWrapper.SmartWrapper.Bindings],
        _run_manager: CallbackManagerForChainRun,
    ) -> str:
        """
        Returns the query results as string
        """

        if isinstance(query_results, str):
            return query_results

        res = [{k: v.value for k, v in d.items()} for d in query_results.bindings]
        query_results_str = json.dumps(res)
        _run_manager.on_text(
            "Query results:",
            color="green",
            end="\n",
            verbose=self.verbose,
        )
        _run_manager.on_text(
            query_results_str,
            color="green",
            end="\n",
            verbose=self.verbose,
        )
        return query_results_str
