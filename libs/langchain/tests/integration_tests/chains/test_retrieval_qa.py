"""Test RetrievalQA functionality."""
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.chains.loading import load_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def test_retrieval_qa_saving_loading(tmp_path: Path) -> None:
    """Test saving and loading."""
    loader = TextLoader("docs/extras/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=docsearch.as_retriever())
    qa.run("What did the president say about Ketanji Brown Jackson?")

    file_path = tmp_path / "RetrievalQA_chain.yaml"
    qa.save(file_path=file_path)
    qa_loaded = load_chain(file_path, retriever=docsearch.as_retriever())

    assert qa_loaded == qa
