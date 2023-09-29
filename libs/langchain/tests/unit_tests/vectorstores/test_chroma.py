import uuid

from langchain.embeddings import FakeEmbeddings as Fak
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
import requests
import pytest


def is_api_accessible(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except:
        return False


def batch_support_chroma_version():
    import chromadb
    major, minor, patch = chromadb.__version__.split(".")
    if int(major) == 0 and int(minor) >= 4 and int(patch) >= 10:
        return True
    return False


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(not is_api_accessible('http://localhost:8000/api/v1/heartbeat'), reason='API not accessible')
@pytest.mark.skipif(not batch_support_chroma_version(), reason='ChromaDB version does not support batching')
def test_chroma_large_batch():
    import chromadb
    client = chromadb.HttpClient()
    embedding_function = Fak(size=255)
    col = client.get_or_create_collection("my_collection", embedding_function=embedding_function.embed_documents)
    docs = ["This is a test document"] * (client.max_batch_size + 100)
    Chroma.from_texts(client=client, collection_name=col.name, texts=docs,
                      embedding=embedding_function,
                      ids=[str(uuid.uuid4()) for _ in range(len(docs))])


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(not is_api_accessible('http://localhost:8000/api/v1/heartbeat'), reason='API not accessible')
@pytest.mark.skipif(not batch_support_chroma_version(), reason='ChromaDB version does not support batching')
def test_chroma_large_batch_update():
    import chromadb
    client = chromadb.HttpClient()
    embedding_function = Fak(size=255)
    col = client.get_or_create_collection("my_collection", embedding_function=embedding_function.embed_documents)
    docs = ["This is a test document"] * (client.max_batch_size + 100)
    ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db = Chroma.from_texts(client=client, collection_name=col.name, texts=docs,
                           embedding=embedding_function,
                           ids=ids)
    new_docs = [Document(page_content="This is a new test document", metadata={"doc_id": f"{i}"}) for i in
                range(len(docs) - 10)]
    new_ids = [_id for _id in ids[:len(new_docs)]]
    db.update_documents(ids=new_ids, documents=new_docs)


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(not is_api_accessible('http://localhost:8000/api/v1/heartbeat'), reason='API not accessible')
@pytest.mark.skipif(batch_support_chroma_version(), reason='ChromaDB version does not support batching')
def test_chroma_legacy_batching():
    import chromadb
    client = chromadb.HttpClient()
    embedding_function = Fak(size=255)
    col = client.get_or_create_collection("my_collection", embedding_function=embedding_function.embed_documents)
    docs = ["This is a test document"] * 100
    Chroma.from_texts(client=client, collection_name=col.name, texts=docs,
                      embedding=embedding_function,
                      ids=[str(uuid.uuid4()) for _ in range(len(docs))])
