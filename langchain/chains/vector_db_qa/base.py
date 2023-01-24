"""Chain for question-answering against a vector database."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
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
    return_source_documents: bool = False
    """Return the source documents."""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

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

    @classmethod
    def from_chain_type(
        cls, llm: BaseLLM, chain_type: str = "stuff", **kwargs: Any
    ) -> VectorDBQA:
        """Load chain from chain type."""
        combine_documents_chain = load_qa_chain(llm, chain_type=chain_type)
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run similarity search and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = vectordbqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        question = inputs[self.input_key]

        docs = self.vectorstore.similarity_search(
            question, k=self.k, **self.search_kwargs
        )
        answer, _ = self.combine_documents_chain.combine_docs(docs, question=question)

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}
