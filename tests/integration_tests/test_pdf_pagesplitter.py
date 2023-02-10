"""Test splitting with page numbers included."""
import os

from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def test_pdf_pagesplitter() -> None:
    """Test splitting with page numbers included."""
    loader = PagedPDFSplitter(chunk_size=250)
    script_dir = os.path.dirname(__file__)
    splits, metadatas = loader.load_and_split(
        os.path.join(script_dir, "examples/hello.pdf")
    )
    assert "pages" in metadatas[0]
    assert "key" in metadatas[0]
    assert len(splits) == len(metadatas)

    faiss_index = FAISS.from_texts(splits, OpenAIEmbeddings(), metadatas=metadatas)
    docs = faiss_index.similarity_search("Complete this sentence: Hello", k=1)
    assert "Hello World" in docs[0].page_content
