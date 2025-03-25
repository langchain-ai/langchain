"""Vectorstore stubs for the indexing api."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, ConfigDict, Field

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA


def _get_default_text_splitter() -> TextSplitter:
    """Return the default text splitter used for chunking documents."""
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class VectorStoreIndexWrapper(BaseModel):
    """Wrapper around a vectorstore for easy access."""

    vectorstore: VectorStore

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def query(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Query the vectorstore using the provided LLM.

        Args:
            question: The question or prompt to query.
            llm: The language model to use. Must not be None.
            retriever_kwargs: Optional keyword arguments for the retriever.
            **kwargs: Additional keyword arguments forwarded to the chain.

        Returns:
            The result string from the RetrievalQA chain.
        """
        if llm is None:
            raise NotImplementedError(
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "llm = OpenAI(temperature=0)"
            )
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return chain.invoke({chain.input_key: question})[chain.output_key]

    async def aquery(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously query the vectorstore using the provided LLM.

        Args:
            question: The question or prompt to query.
            llm: The language model to use. Must not be None.
            retriever_kwargs: Optional keyword arguments for the retriever.
            **kwargs: Additional keyword arguments forwarded to the chain.

        Returns:
            The asynchronous result string from the RetrievalQA chain.
        """
        if llm is None:
            raise NotImplementedError(
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "llm = OpenAI(temperature=0)"
            )
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return (await chain.ainvoke({chain.input_key: question}))[chain.output_key]

    def query_with_sources(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Query the vectorstore and retrieve the answer along with sources.

        Args:
            question: The question or prompt to query.
            llm: The language model to use. Must not be None.
            retriever_kwargs: Optional keyword arguments for the retriever.
            **kwargs: Additional keyword arguments forwarded to the chain.

        Returns:
            A dictionary containing the answer and source documents.
        """
        if llm is None:
            raise NotImplementedError(
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "llm = OpenAI(temperature=0)"
            )
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return chain.invoke({chain.question_key: question})

    async def aquery_with_sources(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously query the vectorstore and retrieve the answer and sources.

        Args:
            question: The question or prompt to query.
            llm: The language model to use. Must not be None.
            retriever_kwargs: Optional keyword arguments for the retriever.
            **kwargs: Additional keyword arguments forwarded to the chain.

        Returns:
            A dictionary containing the answer and source documents.
        """
        if llm is None:
            raise NotImplementedError(
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "llm = OpenAI(temperature=0)"
            )
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return await chain.ainvoke({chain.question_key: question})


def _get_in_memory_vectorstore() -> Type[VectorStore]:
    """Get the InMemoryVectorStore."""
    import warnings

    try:
        from langchain_community.vectorstores.inmemory import InMemoryVectorStore
    except ImportError:
        raise ImportError(
            "Please install langchain-community to use the InMemoryVectorStore."
        )
    warnings.warn(
        "Using InMemoryVectorStore as the default vectorstore."
        "This memory store won't persist data. You should explicitly"
        "specify a vectorstore when using VectorstoreIndexCreator"
    )
    return InMemoryVectorStore


class VectorstoreIndexCreator(BaseModel):
    """Logic for creating indexes."""

    vectorstore_cls: Type[VectorStore] = Field(
        default_factory=_get_in_memory_vectorstore
    )
    embedding: Embeddings
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)
    vectorstore_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from a list of loaders.

        Args:
            loaders: A list of `BaseLoader` instances to load documents.

        Returns:
            A `VectorStoreIndexWrapper` containing the constructed vectorstore.
        """
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return self.from_documents(docs)

    async def afrom_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Asynchronously create a vectorstore index from a list of loaders.

        Args:
            loaders: A list of `BaseLoader` instances to load documents.

        Returns:
            A `VectorStoreIndexWrapper` containing the constructed vectorstore.
        """
        docs = []
        for loader in loaders:
            async for doc in loader.alazy_load():
                docs.append(doc)
        return await self.afrom_documents(docs)

    def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from a list of documents.

        Args:
            documents: A list of `Document` objects.

        Returns:
            A `VectorStoreIndexWrapper` containing the constructed vectorstore.
        """
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)

    async def afrom_documents(
        self, documents: List[Document]
    ) -> VectorStoreIndexWrapper:
        """Asynchronously create a vectorstore index from a list of documents.

        Args:
            documents: A list of `Document` objects.

        Returns:
            A `VectorStoreIndexWrapper` containing the constructed vectorstore.
        """
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = await self.vectorstore_cls.afrom_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
