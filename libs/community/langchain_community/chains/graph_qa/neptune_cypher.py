from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field

from langchain_community.chains.graph_qa.prompts import (
    CYPHER_QA_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT,
)
from langchain_community.graphs import BaseNeptuneGraph

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def trim_query(query: str) -> str:
    """Trim the query to only include Cypher keywords."""
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


@deprecated(
    since="0.3.15",
    removal="1.0",
    alternative_import="langchain_aws.create_neptune_opencypher_qa_chain",
)
class NeptuneOpenCypherQAChain(Chain):
    """Chain for question-answering against a Neptune graph
    by generating openCypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.

    Example:
        .. code-block:: python

        chain = NeptuneOpenCypherQAChain.from_llm(
            llm=llm,
            graph=graph
        )
        response = chain.run(query)
    """

    graph: BaseNeptuneGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""
    extra_instructions: Optional[str] = None
    """Extra instructions by the appended to the query generation prompt."""

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
        extra_instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> NeptuneOpenCypherQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

        _cypher_prompt = cypher_prompt or PROMPT_SELECTOR.get_prompt(llm)
        cypher_generation_chain = LLMChain(llm=llm, prompt=_cypher_prompt)

        return cls(
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            extra_instructions=extra_instructions,
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
            {
                "question": question,
                "schema": self.graph.get_schema,
                "extra_instructions": self.extra_instructions or "",
            },
            callbacks=callbacks,
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
