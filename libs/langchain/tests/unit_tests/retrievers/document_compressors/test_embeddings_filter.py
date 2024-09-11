from langchain_core.embeddings import FakeEmbeddings

from langchain.retrievers.document_compressors import EmbeddingsFilter


def test_embeddings_filter_init() -> None:
    EmbeddingsFilter(embeddings=FakeEmbeddings(size=2))
