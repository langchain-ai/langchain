"""Question answering over a graph."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Field

from langchain_community.chains.graph_qa.prompts import (
    CYPHER_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_TEMPLATE,
    GREMLIN_GENERATION_PROMPT,
)
from langchain_community.graphs import GremlinGraph

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def extract_gremlin(text: str) -> str:
    """Extract Gremlin code from a text.

    Args:
        text: Text to extract Gremlin code from.

    Returns:
        Gremlin code extracted from the text.
    """
    text = text.replace("`", "")
    if text.startswith("gremlin"):
        text = text[len("gremlin") :]
    return text.replace("\n", "")


class GremlinQAChain(Chain):
    """Chain for question-answering against a graph by generating gremlin statements.

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

    graph: GremlinGraph = Field(exclude=True)
    gremlin_generation_chain: LLMChain
    qa_chain: LLMChain
    gremlin_fix_chain: LLMChain
    max_fix_retries: int = 3
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 100
    return_direct: bool = False
    return_intermediate_steps: bool = False

    @property
    def input_keys(self) -> List[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        gremlin_fix_prompt: BasePromptTemplate = PromptTemplate(
            input_variables=["error_message", "generated_sparql", "schema"],
            template=GRAPHDB_SPARQL_FIX_TEMPLATE.replace("SPARQL", "Gremlin").replace(
                "in Turtle format", ""
            ),
        ),
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        gremlin_prompt: BasePromptTemplate = GREMLIN_GENERATION_PROMPT,
        **kwargs: Any,
    ) -> GremlinQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        gremlin_generation_chain = LLMChain(llm=llm, prompt=gremlin_prompt)
        gremlinl_fix_chain = LLMChain(llm=llm, prompt=gremlin_fix_prompt)
        return cls(
            qa_chain=qa_chain,
            gremlin_generation_chain=gremlin_generation_chain,
            gremlin_fix_chain=gremlinl_fix_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Generate gremlin statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        intermediate_steps: List = []

        chain_response = self.gremlin_generation_chain.invoke(
            {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        generated_gremlin = extract_gremlin(
            chain_response[self.gremlin_generation_chain.output_key]
        )

        _run_manager.on_text("Generated gremlin:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_gremlin, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_gremlin})

        if generated_gremlin:
            context = self.execute_with_retry(
                _run_manager, callbacks, generated_gremlin
            )[: self.top_k]
        else:
            context = []

        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})

            result = self.qa_chain.invoke(
                {"question": question, "context": context},
                callbacks=callbacks,
            )
            final_result = result[self.qa_chain.output_key]

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result

    def execute_query(self, query: str) -> List[Any]:
        try:
            return self.graph.query(query)
        except Exception as e:
            if hasattr(e, "status_message"):
                raise ValueError(e.status_message)
            else:
                raise ValueError(str(e))

    def execute_with_retry(
        self,
        _run_manager: CallbackManagerForChainRun,
        callbacks: CallbackManager,
        generated_gremlin: str,
    ) -> List[Any]:
        try:
            return self.execute_query(generated_gremlin)
        except Exception as e:
            retries = 0
            error_message = str(e)
            self.log_invalid_query(_run_manager, generated_gremlin, error_message)

            while retries < self.max_fix_retries:
                try:
                    fix_chain_result = self.gremlin_fix_chain.invoke(
                        {
                            "error_message": error_message,
                            # we are borrowing template from sparql
                            "generated_sparql": generated_gremlin,
                            "schema": self.schema,
                        },
                        callbacks=callbacks,
                    )
                    fixed_gremlin = fix_chain_result[self.gremlin_fix_chain.output_key]
                    return self.execute_query(fixed_gremlin)
                except Exception as e:
                    retries += 1
                    parse_exception = str(e)
                    self.log_invalid_query(_run_manager, fixed_gremlin, parse_exception)

        raise ValueError("The generated Gremlin query is invalid.")

    def log_invalid_query(
        self,
        _run_manager: CallbackManagerForChainRun,
        generated_query: str,
        error_message: str,
    ) -> None:
        _run_manager.on_text("Invalid Gremlin query: ", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_query, color="red", end="\n", verbose=self.verbose
        )
        _run_manager.on_text(
            "Gremlin Query Parse Error: ", end="\n", verbose=self.verbose
        )
        _run_manager.on_text(
            error_message, color="red", end="\n\n", verbose=self.verbose
        )
