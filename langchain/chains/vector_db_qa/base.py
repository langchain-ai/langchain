"""Chain for question-answering against a vector database."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.vector_db_qa.prompt import PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore


class VectorDBQA(Chain, BaseModel):
    """Chain for question-answering against a vector database.

    Example:
        .. code-block:: python

            from langchain import OpenAI, VectorDBQA
            from langchain.faiss import FAISS
            vectordb = FAISS(...)
            vectordbQA = VectorDBQA(llm=OpenAI(), vectorstore=vectordb)

    """

    vectorstore: VectorStore
    """Vector Database to connect to."""
    k: int = 4
    """Number of documents to query for."""
    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    # TODO: deprecate this
    @root_validator(pre=True)
    def load_combine_documents_chain(cls, values: Dict) -> Dict:
        """Validate question chain."""
        if "combine_documents_chain" not in values:
            if "llm" not in values:
                raise ValueError(
                    "If `combine_documents_chain` not provided, `llm` should be."
                )
            prompt = values.pop("prompt", PROMPT)
            llm = values.pop("llm")
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            document_prompt = PromptTemplate(
                input_variables=["page_content"], template="Context:\n{page_content}"
            )
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                document_prompt=document_prompt,
            )
            values["combine_documents_chain"] = combine_documents_chain
        return values

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> VectorDBQA:
        """Initialize from LLM."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="Context:\n{page_content}"
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs[self.input_key]
        docs = self.vectorstore.similarity_search(question, k=self.k)
        answer = self.combine_documents_chain.combine_docs(docs, question=question)
        return {self.output_key: answer}
