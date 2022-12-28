"""Question answering with sources over documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate, RegexParser


class BaseQAWithSourcesChain(Chain, BaseModel, ABC):
    """Question answering with sources over documents."""

    combine_document_chain: BaseCombineDocumentsChain
    """Chain to use to combine documents."""
    question_key: str = "question"  #: :meta private:
    input_docs_key: str = "docs"  #: :meta private:

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            combine_document_chain=combine_results_chain,
            document_variable_name="context",
        )
        return cls(
            combine_document_chain=combine_document_chain,
            **kwargs,
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
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        output_parser = self.combine_document_chain.output_parser
        if not isinstance(output_parser, RegexParser):
            raise ValueError(
                "Output parser of combine_document_chain should be a RegexParser,"
                f" got {output_parser}"
            )
        return output_parser.output_keys

    @root_validator(pre=True)
    def validate_question_chain(cls, values: Dict) -> Dict:
        """Validate question chain."""
        llm_question_chain = values["combine_document_chain"].llm_chain
        if len(llm_question_chain.input_keys) != 2:
            raise ValueError(
                f"The llm_question_chain should have two inputs: a content key "
                f"(the first one) and a question key (the second one). Got "
                f"{llm_question_chain.input_keys}."
            )
        return values

    @root_validator()
    def validate_combine_chain_output(cls, values: Dict) -> Dict:
        """Validate that the combine chain outputs a dictionary."""
        combine_docs_chain = values["combine_document_chain"]
        if not isinstance(combine_docs_chain.output_parser, RegexParser):
            raise ValueError(
                "Output parser of combine_document_chain should be a RegexParser,"
                f" got {combine_docs_chain.output_parser}"
            )

        return values

    @abstractmethod
    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        """Get docs to run questioning over."""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = self._get_docs(inputs)
        answer, _ = self.combine_document_chain.combine_and_parse(docs, **inputs)
        return answer


class QAWithSourcesChain(BaseQAWithSourcesChain, BaseModel):
    """Question answering with sources over documents."""

    input_docs_key: str = "docs"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_docs_key, self.question_key]

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        return inputs.pop(self.input_docs_key)
