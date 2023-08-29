"""Integration test for compression pipelines."""
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


def test_document_compressor_pipeline() -> None:
    embeddings = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_filter = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    texts = [
        "This sentence is about cows",
        "This sentence was about cows",
        "foo bar baz",
    ]
    docs = [Document(page_content=". ".join(texts))]
    actual = pipeline_filter.compress_documents(docs, "Tell me about farm animals")
    assert len(actual) == 1
    assert actual[0].page_content in texts[:2]
