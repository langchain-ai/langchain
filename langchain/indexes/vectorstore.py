from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders.base import BaseLoader
from pydantic import BaseModel, Field, Extra
from typing import List, Type

def _get_default_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class VectorstoreIndexCreator(BaseModel):

    vectorstore_cls: Type[VectorStore] = Chroma
    embedding: Embeddings = Field(default_factory=OpenAIEmbeddings)
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def from_loaders(self, loaders: List[BaseLoader]):
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        sub_docs = self.text_splitter.split_documents(docs)
        vectorstore = self.vectorstore_cls.from_documents(sub_docs, self.embedding)
        return vectorstore
