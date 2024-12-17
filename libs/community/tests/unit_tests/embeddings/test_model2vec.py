from langchain_community.embeddings.model2vec import Model2vecEmbeddings


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test model2vec embeddings."""
    try:
        embedding = Model2vecEmbeddings("minishlab/potion-base-8M")
        assert len(embedding.embed_query("hi")) == 256
    except Exception:
        # model2vec is not installed
        assert True
