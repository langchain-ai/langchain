"""Test RetrievalQA functionality."""
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain.chains.loading import load_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents.context_builder import ContextBuilder
from langchain.docstore.document import Document
from langchain.prompts.base import BasePromptTemplate

class SourceContextBuilder(ContextBuilder):
    def create_context(
            self, docs: List[Document], document_prompt: BasePromptTemplate
    ) -> str:
       return ",".join([doc.metadata["source"] for doc in docs])


def test_retrieval_qa_saving_loading(tmp_path: Path) -> None:
    """Test saving and loading."""
    loader = TextLoader("docs/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type_kwargs={"context_builder": SourceContextBuilder()},
                                     retriever=docsearch.as_retriever())

    qa.combine_documents_chain.llm_chain.verbose = True
    qa("What is the capital of Ireland?")

