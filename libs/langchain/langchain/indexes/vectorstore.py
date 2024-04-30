from typing import Any, Dict, List, Optional, Type

from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA


def _get_default_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class VectorStoreIndexWrapper(BaseModel):
    """Wrapper around a vectorstore for easy access."""

    vectorstore: VectorStore

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def query(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Query the vectorstore."""
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
        """Query the vectorstore."""
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
        """Query the vectorstore and get back sources."""
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
        """Query the vectorstore and get back sources."""
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


class VectorstoreIndexCreator(BaseModel):
    """Logic for creating indexes."""

    vectorstore_cls: Type[VectorStore] = InMemoryVectorStore
    embedding: Embeddings
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)
    vectorstore_kwargs: dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from loaders."""
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return self.from_documents(docs)

    async def afrom_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from loaders."""
        docs = []
        for loader in loaders:
            async for doc in loader.alazy_load():
                docs.append(doc)
        return await self.afrom_documents(docs)

    def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from documents."""
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)

    async def afrom_documents(
        self, documents: List[Document]
    ) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from documents."""
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = await self.vectorstore_cls.afrom_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
