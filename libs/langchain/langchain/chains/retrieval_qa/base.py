"""Chain for question-answering against a vector database."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, model_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This class is deprecated. Use the `create_retrieval_chain` constructor "
        "instead. See migration guide here: "
        "https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/"
    ),
)
class BaseRetrievalQA(Chain):
    """Base class for question-answering chains."""

    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_source_documents: bool = False
    """Return the source documents or not."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = [*_output_keys, "source_documents"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        llm_chain_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            callbacks=callbacks,
            **(llm_chain_kwargs or {}),
        )
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="Context:\n{page_content}",
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            **_chain_type_kwargs,
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @abstractmethod
    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get documents to do question answering over."""

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)  # type: ignore[call-arg]
        answer = self.combine_documents_chain.run(
            input_documents=docs,
            question=question,
            callbacks=_run_manager.get_child(),
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        return {self.output_key: answer}

    @abstractmethod
    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get documents to do question answering over."""

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(question, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(question)  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs,
            question=question,
            callbacks=_run_manager.get_child(),
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        return {self.output_key: answer}


@deprecated(
    since="0.1.17",
    removal="1.0",
    message=(
        "This class is deprecated. Use the `create_retrieval_chain` constructor "
        "instead. See migration guide here: "
        "https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/"
    ),
)
class RetrievalQA(BaseRetrievalQA):
    """Chain for question-answering against an index.

    This class is deprecated. See below for an example implementation using
    `create_retrieval_chain`:

        .. code-block:: python

            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI


            retriever = ...  # Your retriever
            llm = ChatOpenAI()

            system_prompt = (
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. "
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)

            chain.invoke({"input": query})

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAI
            from langchain.chains import RetrievalQA
            from langchain_community.vectorstores import FAISS
            from langchain_core.vectorstores import VectorStoreRetriever
            retriever = VectorStoreRetriever(vectorstore=FAISS(...))
            retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

    """

    retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        return self.retriever.invoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        return await self.retriever.ainvoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retrieval_qa"


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This class is deprecated. Use the `create_retrieval_chain` constructor "
        "instead. See migration guide here: "
        "https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/"
    ),
)
class VectorDBQA(BaseRetrievalQA):
    """Chain for question-answering against a vector database."""

    vectorstore: VectorStore = Field(exclude=True, alias="vectorstore")
    """Vector Database to connect to."""
    k: int = 4
    """Number of documents to query for."""
    search_type: str = "similarity"
    """Search type to use over vectorstore. `similarity` or `mmr`."""
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict) -> Any:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                msg = f"search_type of {search_type} not allowed."
                raise ValueError(msg)
        return values

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                question,
                k=self.k,
                **self.search_kwargs,
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=self.k,
                **self.search_kwargs,
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        msg = "VectorDBQA does not support async"
        raise NotImplementedError(msg)

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "vector_db_qa"
