from langchain_community.embeddings.ovhcloud import OVHCloudEmbeddings


def test_ovhcloud_embed_documents() -> None:
    llm = OVHCloudEmbeddings(model_name="multilingual-e5-base")
    docs = ["Hello", "World"]
    output = llm.embed_documents(docs)
    assert len(output) == len(docs)
