from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
    CYPHER_QA_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.graphs import NeptuneGraph
from langchain.prompts.base import BasePromptTemplate
from langchain.pydantic_v1 import Field

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def trim_query(query: str) -> str:
    keywords = (
        "CALL",
        "CREATE",
        "DELETE",
        "DETACH",
        "LIMIT",
        "MATCH",
        "MERGE",
        "OPTIONAL",
        "ORDER",
        "REMOVE",
        "RETURN",
        "SET",
        "SKIP",
        "UNWIND",
        "WITH",
        "WHERE",
        "//",
    )

    lines = query.split("\n")
    new_query = ""

    for line in lines:
        if line.strip().upper().startswith(keywords):
            new_query += line + "\n"

    return new_query


def extract_cypher(text: str) -> str:
    """Extract Cypher code from text using Regex."""
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


def use_simple_prompt(llm: BaseLanguageModel) -> bool:
    """Decides whether to use the simple prompt"""
    if llm._llm_type and "anthropic" in llm._llm_type:  # type: ignore
        return True

    # Bedrock anthropic
    if hasattr(llm, "model_id") and "anthropic" in llm.model_id:  # type: ignore
        return True

    return False


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=NEPTUNE_OPENCYPHER_GENERATION_PROMPT,
    conditionals=[(use_simple_prompt, NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT)],
)


class NeptuneOpenCypherQAChain(Chain):
    """Chain for question-answering against a Neptune graph
    by generating openCypher statements.

    Example:
        .. code-block:: python

        chain = NeptuneOpenCypherQAChain.from_llm(
            llm=llm,
            graph=graph
        )
        response = chain.run(query)
    """

    graph: NeptuneGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
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
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        cypher_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> NeptuneOpenCypherQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

        _cypher_prompt = cypher_prompt or PROMPT_SELECTOR.get_prompt(llm)
        cypher_generation_chain = LLMChain(llm=llm, prompt=_cypher_prompt)

        return cls(
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
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
            {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)
        generated_cypher = trim_query(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        context = self.graph.query(generated_cypher)

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
