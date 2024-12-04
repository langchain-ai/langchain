from langchain_community.embeddings.model2vec import Model2vecEmbeddings


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test model2vec embeddings."""
    embedding = Model2vecEmbeddings()
    assert len(embedding.embed_query("hi")) == 256
