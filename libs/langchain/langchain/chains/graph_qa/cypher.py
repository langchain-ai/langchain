"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str],
    exclude_types: List[str],
) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }

    return (
        f"Node properties are the following: \n {filtered_schema['node_props']}\n"
        f"Relationships properties are the following: \n {filtered_schema['rel_props']}"
        "\nRelationships are: \n"
        + str(
            [
                f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
                for el in filtered_schema["relationships"]
            ]
        )
    )


class GraphCypherQAChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements."""

    graph: Neo4jGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    graph_schema: str
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""
    cypher_query_corrector: Optional[CypherQueryCorrector] = None
    """Optional cypher validation tool"""

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

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        validate_cypher: bool = False,
        **kwargs: Any,
    ) -> GraphCypherQAChain:
        """Initialize from LLM."""

        if not cypher_llm and not llm:
            raise ValueError("Either `llm` or `cypher_llm` parameters must be provided")
        if not qa_llm and not llm:
            raise ValueError("Either `llm` or `qa_llm` parameters must be provided")
        if cypher_llm and qa_llm and llm:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'"
                ", and 'llm', but not all three simultaneously."
            )

        qa_chain = LLMChain(llm=qa_llm or llm, prompt=qa_prompt)
        cypher_generation_chain = LLMChain(llm=cypher_llm or llm, prompt=cypher_prompt)

        if exclude_types and include_types:
            raise ValueError(
                "Either `exclude_types` or `include_types` "
                "can be provided, but not both"
            )

        graph_schema = construct_schema(
            kwargs["graph"].structured_schema, include_types, exclude_types
        )

        cypher_query_corrector = None
        if validate_cypher:
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in kwargs["graph"].structured_schema.get("relationships")
            ]
            cypher_query_corrector = CypherQueryCorrector(corrector_schema)

        return cls(
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            cypher_query_corrector=cypher_query_corrector,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.run(
            {"question": question, "schema": self.graph_schema}, callbacks=callbacks
        )

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
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

            result = self.qa_chain(
                {"question": question, "context": context},
                callbacks=callbacks,
            )
            final_result = result[self.qa_chain.output_key]

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
