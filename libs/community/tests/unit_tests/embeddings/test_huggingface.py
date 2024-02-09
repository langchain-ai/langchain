from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test huggingface embeddings."""
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key="abcd123")
    assert "abcd123" not in repr(embedding)
