import hashlib

def generate_namespaced_id(text: str, namespace: str) -> str:
    """Generate a unique document ID scoped to a collection."""
    hash_base = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{namespace}_{hash_base}"


def test_same_doc_in_multiple_collections_has_unique_ids():
    from langchain_community.vectorstores.chroma import Chroma
    from langchain_community.embeddings import FakeEmbeddings

    texts = ["AI agents are evolving."]

    db1 = Chroma(collection_name="alpha", embedding_function=FakeEmbeddings())
    db2 = Chroma(collection_name="beta", embedding_function=FakeEmbeddings())

    db1.add_texts(texts)
    db2.add_texts(texts)

    result1 = db1.similarity_search("AI agents", k=1)
    result2 = db2.similarity_search("AI agents", k=1)

    assert result1[0].page_content == result2[0].page_content
