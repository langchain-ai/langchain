"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from langchain_community.chains.graph_qa.prompts import (
    AQL_FIX_PROMPT,
    AQL_GENERATION_PROMPT,
    AQL_QA_PROMPT,
)
from langchain_community.graphs.arangodb_graph import ArangoGraph


class ArangoGraphQAChain(Chain):
    """Chain for question-answering against a graph by generating AQL statements.

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

    graph: ArangoGraph = Field(exclude=True)
    aql_generation_chain: LLMChain
    aql_fix_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    # Specifies the maximum number of AQL Query Results to return
    top_k: int = 10

    # Specifies the set of AQL Query Examples that promote few-shot-learning
    aql_examples: str = ""

    # Specify whether to return the AQL Query in the output dictionary
    return_aql_query: bool = False

    # Specify whether to return the AQL JSON Result in the output dictionary
    return_aql_result: bool = False

    # Specify the maximum amount of AQL Generation attempts that should be made
    max_aql_generation_attempts: int = 3

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
        aql_generation_prompt: BasePromptTemplate = AQL_GENERATION_PROMPT,
        aql_fix_prompt: BasePromptTemplate = AQL_FIX_PROMPT,
        **kwargs: Any,
    ) -> ArangoGraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        aql_generation_chain = LLMChain(llm=llm, prompt=aql_generation_prompt)
        aql_fix_chain = LLMChain(llm=llm, prompt=aql_fix_prompt)

        return cls(
            qa_chain=qa_chain,
            aql_generation_chain=aql_generation_chain,
            aql_fix_chain=aql_fix_chain,
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

        Users can modify the following ArangoGraphQAChain Class Variables:

        :var top_k: The maximum number of AQL Query Results to return
        :type top_k: int

        :var aql_examples: A set of AQL Query Examples that are passed to
            the AQL Generation Prompt Template to promote few-shot-learning.
            Defaults to an empty string.
        :type aql_examples: str

        :var return_aql_query: Whether to return the AQL Query in the
            output dictionary. Defaults to False.
        :type return_aql_query: bool

        :var return_aql_result: Whether to return the AQL Query in the
            output dictionary. Defaults to False
        :type return_aql_result: bool

        :var max_aql_generation_attempts: The maximum amount of AQL
            Generation attempts to be made prior to raising the last
            AQL Query Execution Error. Defaults to 3.
        :type max_aql_generation_attempts: int
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        user_input = inputs[self.input_key]

        #########################
        # Generate AQL Query #
        aql_generation_output = self.aql_generation_chain.run(
            {
                "adb_schema": self.graph.schema,
                "aql_examples": self.aql_examples,
                "user_input": user_input,
            },
            callbacks=callbacks,
        )
        #########################

        aql_query = ""
        aql_error = ""
        aql_result = None
        aql_generation_attempt = 1

        while (
            aql_result is None
            and aql_generation_attempt < self.max_aql_generation_attempts + 1
        ):
            #####################
            # Extract AQL Query #
            pattern = r"```(?i:aql)?(.*?)```"
            matches = re.findall(pattern, aql_generation_output, re.DOTALL)
            if not matches:
                _run_manager.on_text(
                    "Invalid Response: ", end="\n", verbose=self.verbose
                )
                _run_manager.on_text(
                    aql_generation_output, color="red", end="\n", verbose=self.verbose
                )
                raise ValueError(f"Response is Invalid: {aql_generation_output}")

            aql_query = matches[0]
            #####################

            _run_manager.on_text(
                f"AQL Query ({aql_generation_attempt}):", verbose=self.verbose
            )
            _run_manager.on_text(
                aql_query, color="green", end="\n", verbose=self.verbose
            )

            #####################
            # Execute AQL Query #
            from arango import AQLQueryExecuteError

            try:
                aql_result = self.graph.query(aql_query, self.top_k)
            except AQLQueryExecuteError as e:
                aql_error = e.error_message

                _run_manager.on_text(
                    "AQL Query Execution Error: ", end="\n", verbose=self.verbose
                )
                _run_manager.on_text(
                    aql_error, color="yellow", end="\n\n", verbose=self.verbose
                )

                ########################
                # Retry AQL Generation #
                aql_generation_output = self.aql_fix_chain.run(
                    {
                        "adb_schema": self.graph.schema,
                        "aql_query": aql_query,
                        "aql_error": aql_error,
                    },
                    callbacks=callbacks,
                )
                ########################

            #####################

            aql_generation_attempt += 1

        if aql_result is None:
            m = f"""
                Maximum amount of AQL Query Generation attempts reached.
                Unable to execute the AQL Query due to the following error:
                {aql_error}
            """
            raise ValueError(m)

        _run_manager.on_text("AQL Result:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(aql_result), color="green", end="\n", verbose=self.verbose
        )

        ########################
        # Interpret AQL Result #
        result = self.qa_chain(
            {
                "adb_schema": self.graph.schema,
                "user_input": user_input,
                "aql_query": aql_query,
                "aql_result": aql_result,
            },
            callbacks=callbacks,
        )
        ########################

        # Return results #
        result = {self.output_key: result[self.qa_chain.output_key]}

        if self.return_aql_query:
            result["aql_query"] = aql_query

        if self.return_aql_result:
            result["aql_result"] = aql_result

        return result
