from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_filters import EmbeddingRedundantDocumentFilter
from langchain.vectorstores import Chroma


def test_contextual_compression_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts = [
        "This is a document about the Boston Celtics",
        "This document is about the Boston Celtics",
        "The Boston Celtics won the game by 20 points",
    ]
    embeddings = OpenAIEmbeddings()
    base_filter = EmbeddingRedundantDocumentFilter(embeddings=embeddings)
    base_retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": 3}
    )
    retriever = ContextualCompressionRetriever(
        base_filter=base_filter, base_retriever=base_retriever
    )

    actual = retriever.get_relevant_documents("Tell me about the Celtics")
    assert len(actual) == 2
    assert set([d.page_content for d in actual]).difference(texts[:2])
