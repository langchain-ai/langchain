from langchain_community.embeddings.xinference import XinferenceEmbeddings


def test_xinference_embedding() -> None:
    embedding_model = XinferenceEmbeddings(
        server_url="http://xinference-hostname:9997", model_uid="foo"
    )

    data = embedding_model.embed_documents(
        texts=["hello", "i'm trying to upgrade xinference embedding"]
    )
