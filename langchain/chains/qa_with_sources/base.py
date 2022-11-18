"""Question answering with sources over documents."""

from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt
from langchain.prompts.prompt import Prompt
from langchain.text_splitter import TextSplitter
from langchain.chains.combine_documents import CombineDocumentsChain
from langchain.chains.qa_with_sources.prompt import example_prompt, combine_prompt, question_prompt

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
    output_key: str = "answer"  #: :meta private:

    @classmethod
    def from_llm(cls, llm: LLM, **kwargs: Any):
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        return cls(llm_question_chain=llm_question_chain, llm_combine_chain=llm_combine_chain, **kwargs)


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_docs_key]
        doc_dicts = [{k: doc.dict()[k] for k in self.document_prompt.input_variables} for doc in inputs[self.input_key]]
        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        doc_variable = self.llm_chain.prompt.input_variables[0]
        output = self.llm_chain.predict(**{doc_variable: "\n\n".join(doc_strings)})
        return {self.output_key: output}
