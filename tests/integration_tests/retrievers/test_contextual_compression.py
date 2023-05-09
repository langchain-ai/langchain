from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.vectorstores import Chroma


def test_contextual_compression_retriever_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts = [
        "This is a document about the Boston Celtics",
        "The Boston Celtics won the game by 20 points",
        "I simply love going to the movies",
    ]
    embeddings = OpenAIEmbeddings()
    base_compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    base_retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": len(texts)}
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=base_compressor, base_retriever=base_retriever
    )

    actual = retriever.get_relevant_documents("Tell me about the Celtics")
    assert len(actual) == 2
    assert texts[-1] not in [d.page_content for d in actual]
