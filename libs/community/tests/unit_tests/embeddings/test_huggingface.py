from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEncoderEmbeddings


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test huggingface embeddings."""
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key="abcd123")
    assert "abcd123" not in repr(embedding)


def test_huggingface_encoder_embeddings_init_and_embed() -> None:
    embedding = HuggingFaceEncoderEmbeddings(
        model_name="distilbert/distilbert-base-uncased",
        tokenizer_name="distilbert/distilbert-base-uncased",
        device="cpu", # or device="cuda:0",
        batch_size=2,
        use_cls_embedding=False,
        model_kwargs = {},
        tokenizer_kwargs={
            "max_length": 768, 
            "add_special_tokens": False
        }
    )
    fake_doc_1 = "hello world"
    fake_doc_2 = "I'm langchaie"

    batch_embeddings = embedding.embed_documents(
        [
            fake_doc_1, fake_doc_2, 
            fake_doc_1, fake_doc_1, 
            fake_doc_2, 
        ]
    )
    embedding1 = embedding.embed_query(fake_doc_1) 
    embedding2 = embedding.embed_query(fake_doc_2)
    
    round_to = 2
    for i, entry in enumerate(embedding1):
        assert(round(entry, round_to) == round(batch_embeddings[0][i], round_to))
        assert(round(entry, round_to) == round(batch_embeddings[2][i], round_to))
        assert(round(entry, round_to) == round(batch_embeddings[3][i], round_to))
    for i, entry in enumerate(embedding2):
        assert(round(entry, round_to) == round(batch_embeddings[1][i], round_to))
        assert(round(entry, round_to) == round(batch_embeddings[4][i], round_to))
