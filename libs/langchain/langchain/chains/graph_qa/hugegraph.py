"""Question answering over a graph."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
    CYPHER_QA_PROMPT,
    GREMLIN_GENERATION_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.graphs.hugegraph import HugeGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class HugeGraphQAChain(Chain):
    """Chain for question-answering against a graph by generating gremlin statements."""

    graph: HugeGraph = Field(exclude=True)
    gremlin_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

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
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        gremlin_prompt: BasePromptTemplate = GREMLIN_GENERATION_PROMPT,
        **kwargs: Any,
    ) -> HugeGraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        gremlin_generation_chain = LLMChain(llm=llm, prompt=gremlin_prompt)

        return cls(
            qa_chain=qa_chain,
            gremlin_generation_chain=gremlin_generation_chain,
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

        generated_gremlin = self.gremlin_generation_chain.run(
            {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        _run_manager.on_text("Generated gremlin:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_gremlin, color="green", end="\n", verbose=self.verbose
        )
        context = self.graph.query(generated_gremlin)

        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(context), color="green", end="\n", verbose=self.verbose
        )

        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=callbacks,
        )
        return {self.output_key: result[self.qa_chain.output_key]}
