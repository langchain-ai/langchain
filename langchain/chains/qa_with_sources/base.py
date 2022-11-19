"""Question answering with sources over documents."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Chain
from langchain.chains.combine_documents import CombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.prompt import (
    combine_prompt,
    example_prompt,
    question_prompt,
)
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt
from langchain.prompts.prompt import Prompt
from langchain.text_splitter import TextSplitter


def _get_default_document_prompt():
    return Prompt(input_variables=["page_content"], template="{page_content}")


class QAWithSourcesChain(Chain, BaseModel):
    """Question answering with sources over documents."""

    llm_question_chain: LLMChain
    """LLM wrapper to use for asking questions to each document."""
    document_prompt: Prompt = example_prompt
    """The Prompt to use to format the response from each document."""
    llm_combine_chain: LLMChain
    """LLM wrapper to use for combining answers."""
    question_key: str = "question"  #: :meta private:
    input_docs_key: str = "docs"  #: :meta private:
    answer_key: str = "answer"  #: :meta private:
    sources_key: str = "sources"  #: :meta private:

    @classmethod
    def from_llm(cls, llm: LLM, **kwargs: Any):
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        return cls(
            llm_question_chain=llm_question_chain,
            llm_combine_chain=llm_combine_chain,
            **kwargs
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_docs_key, self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.answer_key, self.sources_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_docs_key]
        query = inputs[self.question_key]
        content_key, query_key = self.llm_question_chain.input_keys
        results = self.llm_question_chain.apply(
            [{content_key: d.page_content, query_key: query} for d in docs]
        )
        question_result_key = self.llm_question_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            for i, r in enumerate(results)
        ]
        combine_chain = CombineDocumentsChain(
            llm_chain=self.llm_combine_chain, document_prompt=self.document_prompt
        )
        answer_dict = combine_chain(
            {combine_chain.input_key: result_docs, self.question_key: query}
        )
        answer = answer_dict[combine_chain.output_key]
        answer, sources = answer.split("\nSources: ")
        return {self.answer_key: answer, self.sources_key: sources}
