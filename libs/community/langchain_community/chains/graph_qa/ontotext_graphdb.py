from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import SPARQLWrapper
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from langchain_community.chains.graph_qa.prompts import (
    GRAPHDB_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_PROMPT,
    GRAPHDB_SPARQL_GENERATION_PROMPT,
)
from langchain_community.graphs import OntotextGraphDBGraph


class OntotextGraphDBQAChain(Chain):
    """Question-answering against Ontotext GraphDB
    https://graphdb.ontotext.com/ by generating SPARQL queries.
    """

    graph: OntotextGraphDBGraph = Field(exclude=True)
    sparql_generation_chain: LLMChain
    sparql_fix_chain: LLMChain
    max_fix_retries: int
    qa_chain: LLMChain

    input_variables: List[str]
    output_key_answer: str = "answer"  #: :meta private:
    output_key_generated_sparql: str = "generated_sparql"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        _output_keys = [self.output_key_answer, self.output_key_generated_sparql]
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
            sparql_generation_chain=sparql_generation_chain,
            sparql_fix_chain=sparql_fix_chain,
            max_fix_retries=max_fix_retries,
            qa_chain=qa_chain,
            input_variables=sparql_generation_prompt.input_variables,
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

        sparql_generation_chain_result = self.sparql_generation_chain.invoke(
            inputs, callbacks=callbacks
        )
        generated_sparql = sparql_generation_chain_result[
            self.sparql_generation_chain.output_key
        ]

        generated_valid_sparql, query_results = self._execute_query(
            inputs, generated_sparql, _run_manager, callbacks
        )
        query_results_str = self.query_results_to_json(query_results)
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

        inputs.update({"context": query_results_str})
        qa_chain_result = self.qa_chain.invoke(inputs, callbacks=callbacks)
        answer = qa_chain_result[self.qa_chain.output_key]

        return {
            self.output_key_answer: answer,
            self.output_key_generated_sparql: generated_valid_sparql,
        }

    def _execute_query(
        self,
        inputs: Dict[str, Any],
        generated_sparql: str,
        _run_manager: CallbackManagerForChainRun,
        callbacks: CallbackManager,
    ) -> Tuple[
        str,
        Union[
            Union[
                SPARQLWrapper.SmartWrapper.Bindings,
                SPARQLWrapper.SmartWrapper.QueryResult,
            ],
            SPARQLWrapper.SmartWrapper.QueryResult.ConvertResult,
        ],
    ]:
        """
        Executes the generated SPARQL query.
        In case of invalid SPARQL query in terms of syntax,
        try to generate the query again up to a certain number of times.
        """

        from urllib.error import HTTPError

        from SPARQLWrapper.SPARQLExceptions import (
            EndPointInternalError,
            EndPointNotFound,
            QueryBadFormed,
            Unauthorized,
            URITooLong,
        )

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
            return generated_sparql, self.graph.exec_query(generated_sparql)
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
                    inputs.update(
                        {
                            "error_message": error_message,
                            "generated_sparql": generated_sparql,
                        }
                    )
                    sparql_fix_chain_result = self.sparql_fix_chain.invoke(
                        inputs, callbacks=callbacks
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
                    return generated_sparql, self.graph.exec_query(generated_sparql)
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

        except Unauthorized as e:  # status code 401
            _run_manager.on_text(
                f"Unauthorized (status code 401): {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
            raise e
        except EndPointNotFound as e:  # status code 404
            _run_manager.on_text(
                f"EndPointNotFound (status code 404): {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
            raise e
        except URITooLong as e:  # status code 414
            _run_manager.on_text(
                f"URITooLong (status code 414): {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
            raise e
        except EndPointInternalError as e:  # status code 500
            _run_manager.on_text(
                f"EndPointInternalError (status code 500): {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
            raise e
        # code is different to ``400``, ``401``, ``404``, ``414``, ``500``
        except HTTPError as e:
            _run_manager.on_text(
                f"HTTPError (status code {e.status}): {str(e)}",
                color="red",
                end="\n",
                verbose=self.verbose,
            )
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

    @staticmethod
    def query_results_to_json(
        query_results: Union[
            Union[
                SPARQLWrapper.SmartWrapper.Bindings,
                SPARQLWrapper.SmartWrapper.QueryResult,
            ],
            SPARQLWrapper.SmartWrapper.QueryResult.ConvertResult,
        ],
    ) -> str:
        """
        Returns the query results in json format
        """
        query_bindings = query_results.bindings  # List[Dict[str, Value]]
        res = [{k: v.value for k, v in d.items()} for d in query_bindings]
        query_results_str = json.dumps(res)
        return query_results_str
