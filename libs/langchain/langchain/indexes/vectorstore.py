from typing import Any, Dict, List, Optional, Type

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
            raise ValueError("")
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return chain.run(question)

    def query_with_sources(
        self,
        question: str,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Query the vectorstore and get back sources."""
        if llm is None:
            raise ValueError("")
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return chain({chain.question_key: question})


def _embedding_default_factory() -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "Unable to import OpenAIEmbeddings from langchain-openai. Please install "
            "with `pip install -U langchain-openai`."
        ) from e
    return OpenAIEmbeddings()


class VectorstoreIndexCreator(BaseModel):
    """Logic for creating indexes."""

    vectorstore_cls: Type[VectorStore]
    embedding: Embeddings = Field(default_factory=_embedding_default_factory)
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

    def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from documents."""
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
