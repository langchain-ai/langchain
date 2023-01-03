"""Chain for question-answering against a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.vector_db_qa.prompt import PROMPT
from langchain.docstore.document import Document
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

            # DEPRECATED: In favor of the multi-db interface below
            # vectordbQA = VectorDBQA(llm=OpenAI(), vectorstore=vectordb)
            vectordbQA = VectorDBQA(llm=OpenAI(), vectorstores=[vectordb])

            vectordb2 = FAISS(...)
            vectordbQA = VectorDBQA(llm=OpenAI(), vectorstore=[vectordb, vectordb2])
    """

    vectorstore: Optional[VectorStore]  # type: ignore
    """[DEPRECATED] Vector Database to connect to. Use the key `vectorstores` instead."""

    vectorstores: Optional[List[VectorStore]]
    """Vector Databases to connect to."""

    k: int = 4
    """Number of documents to query for."""

    n: int = 4
    """Number of documents to rank from (per vectorstore)."""

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

    @root_validator(pre=True)
    def validate_vectorstores(cls, values: Dict) -> Dict:
        """Validate vectorstores."""

        # Check to make sure vectorstores defined correctly
        only_one_key_defined = bool("vectorstore" in values) ^ bool(
            "vectorstores" in values
        )

        assert (
            only_one_key_defined
        ), "Only one of `vectorstore` or `vectorstores` may be defined."

        return values

    def __init__(self, *args, **kwargs):  # type: ignore

        # Call super to init instance
        super().__init__(*args, **kwargs)

        # Set `vectorstores` correctly
        if self.vectorstore and not self.vectorstores:
            self.vectorstores = [self.vectorstore]
            self.vectorstore = None

        # Make sure cumulative top-n is greater than k
        num_stores = len(self.vectorstores)
        cumulative_n = self.n * num_stores

        assert cumulative_n >= self.k, (
            f"`n={n}` times number of stores={num_stores} "
            f"is smaller than specified `k={self.k}`."
        )

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
        """Call all vectorstores to get relevant documents before searching."""
        question = inputs[self.input_key]

        stores = self.vectorstores or []
        num_stores = len(stores)
        docs: List[Document] = []

        # Get docs from all vectorstores
        for store in stores:
            results = store.similarity_search(question, k=self.n)
            store_k = (self.k // num_stores) + 1

            docs.extend(results[:store_k])

        # Take only top k, assuming results in descending order of relevance
        docs = docs[: self.k]

        # Get answer
        answer, _ = self.combine_documents_chain.combine_docs(docs, question=question)

        return {self.output_key: answer}
