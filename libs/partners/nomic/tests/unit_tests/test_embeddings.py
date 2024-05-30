"""Test embedding model integration."""


from langchain_nomic.embeddings import NomicEmbeddings, NomicMultimodalEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    NomicEmbeddings(model="nomic-embed-text-v1")
    NomicMultimodalEmbeddings(vision_model="nomic-embed-vision-v1", text_model="nomic-embed-text-v1")
