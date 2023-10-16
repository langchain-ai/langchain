"""Test RetrievalQA functionality."""
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.loading import load_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def test_retrieval_qa_with_sources_chain_saving_loading(tmp_path: str) -> None:
    """Test saving and loading."""
    loader = DirectoryLoader("docs/extras/modules/", glob="*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQAWithSourcesChain.from_llm(
        llm=OpenAI(), retriever=docsearch.as_retriever()
    )
    qa("What did the president say about Ketanji Brown Jackson?")

    file_path = tmp_path + "/RetrievalQAWithSourcesChain.yaml"
    qa.save(file_path=file_path)
    qa_loaded = load_chain(file_path, retriever=docsearch.as_retriever())

    assert qa_loaded == qa
