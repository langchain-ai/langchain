from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.dbpedia.prompt import ANSWER_PROMPT_SELECTOR, PROMPT_SELECTOR
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel


class DBPediaChain(Chain):
    query_chain: LLMChain
    answer_chain: LLMChain
    input_key: str = "question"
    output_key: str = "answer"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        query_prompt: Optional[BasePromptTemplate] = None,
        answer_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> DBPediaChain:
        query_prompt = query_prompt or PROMPT_SELECTOR.get_prompt(llm)
        query_chain = LLMChain(llm=llm, prompt=query_prompt)
        answer_prompt = answer_prompt or ANSWER_PROMPT_SELECTOR.get_prompt(llm)
        answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
        return cls(query_chain=query_chain, answer_chain=answer_chain, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        from SPARQLWrapper import JSON, SPARQLWrapper

        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        query = self.query_chain.run(inputs[self.input_key])
        self.callback_manager.on_text("Query written:", end="\n", verbose=self.verbose)
        self.callback_manager.on_text(
            query, color="green", end="\n", verbose=self.verbose
        )
        sparql.setQuery(query)
        result = sparql.query().convert()
        self.callback_manager.on_text(
            "Response gotten:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            result, color="green", end="\n", verbose=self.verbose
        )
        answer = self.answer_chain.run(
            question=inputs[self.input_key], query=query, response=result
        )
        return {self.output_key: answer}
