from typing import Any, List, Optional, Type

from pydantic import BaseModel, Extra, Field

from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.vector_db_qa.base import VectorDBQA
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma


def _get_default_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def _get_default_vectorstore_kwargs() -> dict:
    return dict(embedding=OpenAIEmbeddings())


class VectorStoreIndexWrapper(BaseModel):
    """Wrapper around a vectorstore for easy access."""

    vectorstore: VectorStore

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def query(self, question: str, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        """Query the vectorstore."""
        llm = llm or OpenAI(temperature=0)
        chain = VectorDBQA.from_chain_type(llm, vectorstore=self.vectorstore, **kwargs)
        return chain.run(question)

    def query_with_sources(
        self, question: str, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> dict:
        """Query the vectorstore and get back sources."""
        llm = llm or OpenAI(temperature=0)
        chain = VectorDBQAWithSourcesChain.from_chain_type(
            llm, vectorstore=self.vectorstore, **kwargs
        )
        return chain({chain.question_key: question})


class VectorstoreIndexCreator(BaseModel):
    """Logic for creating indexes."""

    vectorstore_cls: Type[VectorStore] = Chroma
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)
    vectorstore_kwargs: dict = Field(default_factory=_get_default_vectorstore_kwargs)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the class."""
        super().__init__(*args, **kwargs)

        """backwards compatibility for embedding param"""
        if "embedding" in kwargs:
            self.vectorstore_kwargs["embedding"] = kwargs["embedding"]

    def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from loaders."""
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        sub_docs = self.text_splitter.split_documents(docs)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
