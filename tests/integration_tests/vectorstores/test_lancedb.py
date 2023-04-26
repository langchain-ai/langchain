import lancedb
from langchain.vectorstores import LanceDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_lancedb():
    embed = FakeEmbeddings()
    db = lancedb.connect("/tmp/lancedb")
    texts = ["text 1", "text 2", "item 3"]
    vectors = embed.embed_documents(texts)
    table = db.create_table(
        "my_table",
        data=[
            {"vector": vectors[idx], "id": text, "text": text}
            for idx, text in enumerate(texts)
        ],
        mode="overwrite",
    )
    store = LanceDB(table, embedding_function=embed.embed_documents)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


def test_lancedb_add_texts():
    embed = FakeEmbeddings()
    db = lancedb.connect("/tmp/lancedb")
    texts = ["text 1"]
    vectors = embed.embed_documents(texts)
    table = db.create_table(
        "my_table",
        data=[
            {"vector": vectors[idx], "id": text, "text": text}
            for idx, text in enumerate(texts)
        ],
        mode="overwrite",
    )
    store = LanceDB(table, embedding_function=embed.embed_documents)
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts
