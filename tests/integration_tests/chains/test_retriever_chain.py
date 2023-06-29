"""Test Retriever Chain functionality."""
from pathlib import Path

from langchain.chains.loading import load_chain
from langchain.chains.mlflow_retriever_evaluator import MlflowRetrieverEvaluator
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def test_mlflow_retriever_evaluator_saving_loading(tmp_path: Path) -> None:
    """Test saving and loading."""
    loader = TextLoader("docs/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    retriever = MlflowRetrieverEvaluator(retriever=docsearch.as_retriever())

    file_path = tmp_path / "MlflowRetrieverEvaluator.yaml"
    retriever.save(file_path=file_path)
    retriever_loaded = load_chain(file_path, retriever=docsearch.as_retriever())

    assert retriever_loaded == retriever
