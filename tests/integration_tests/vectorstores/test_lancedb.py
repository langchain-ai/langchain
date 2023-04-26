import lancedb
from langchain.vectorstores import LanceDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_lancedb():
    embed = FakeEmbeddings()
    db = lancedb.connect("/tmp/lancedb")
    texts = ["text 1", "text 2", "item 3"]
    table = db.create_table(
        "my_table",
        data=[
            {"vector": embed.embed_documents([text])[0], "id": text, "text": text}
            for text in texts
        ],
        mode="overwrite",
    )
    store = LanceDB(table, embedding_function=embed.embed_documents)
    store.similarity_search("ext")
