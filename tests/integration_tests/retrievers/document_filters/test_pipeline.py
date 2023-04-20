"""Integration test for filtering pipelines."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingRedundantDocumentFilter,
    EmbeddingRelevancyDocumentFilter,
    SplitterDocumentFilter,
)
from langchain.retrievers.document_compressors.base import _RetrievedDocument
from langchain.text_splitter import CharacterTextSplitter


def test_pipeline_filter() -> None:
    embeddings = OpenAIEmbeddings()
    splitter_filter = SplitterDocumentFilter(
        splitter=CharacterTextSplitter(chunk_size=20, chunk_overlap=0, separator=". ")
    )
    redundant_filter = EmbeddingRedundantDocumentFilter(embeddings=embeddings)
    relevant_filter = EmbeddingRelevancyDocumentFilter(
        embeddings=embeddings, similarity_threshold=0.8
    )
    pipeline_filter = DocumentCompressorPipeline(
        compressors=[splitter_filter, redundant_filter, relevant_filter]
    )
    texts = [
        "This sentence is about cows",
        "This sentence was about cows",
        "foo bar baz",
    ]
    docs = [_RetrievedDocument(page_content=". ".join(texts))]
    actual = pipeline_filter.compress_documents(docs, "Tell me about farm animals")
    assert len(actual) == 1
    assert actual[0].page_content in texts[:2]
