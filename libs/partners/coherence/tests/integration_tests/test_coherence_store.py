import inspect
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from coherence import NamedCache, Session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_coherence import CoherenceVectorStore


@pytest_asyncio.fixture
async def store() -> AsyncGenerator[CoherenceVectorStore, None]:
    session: Session = await Session.create()
    named_cache: NamedCache[str, Document] = await session.get_cache("my-map")
    embedding: Embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    cvs: CoherenceVectorStore = await CoherenceVectorStore.create(
        named_cache, embedding
    )
    yield cvs
    await cvs.cache.destroy()
    await session.close()


def get_test_data():
    d1: Document = Document(id="1", page_content="apple")
    d2: Document = Document(id="2", page_content="orange")
    d3: Document = Document(id="3", page_content="tiger")
    d4: Document = Document(id="4", page_content="cat")
    d5: Document = Document(id="5", page_content="dog")
    d6: Document = Document(id="6", page_content="fox")
    d7: Document = Document(id="7", page_content="pear")
    d8: Document = Document(id="8", page_content="banana")
    d9: Document = Document(id="9", page_content="plum")
    d10: Document = Document(id="10", page_content="lion")

    documents = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
    return documents


@pytest.mark.asyncio
async def test_aget_by_id(store: CoherenceVectorStore):
    print()
    print(f"=======: {inspect.currentframe().f_code.co_name}")
    documents = get_test_data()
    await store.aadd_documents(documents)
    ids = [doc.id for doc in documents]
    l = await store.aget_by_ids(ids)
    assert len(l) == 10
    print("====")
    for e in l:
        print(e)


@pytest.mark.asyncio
async def test_adelete(store: CoherenceVectorStore):
    print()
    print(f"=======: {inspect.currentframe().f_code.co_name}")
    documents = get_test_data()
    await store.aadd_documents(documents)
    ids = [doc.id for doc in documents]
    l = await store.aget_by_ids(ids)
    assert len(l) == 10
    await store.adelete(["1", "2"])
    l = await store.aget_by_ids(ids)
    assert len(l) == 8
    await store.adelete()
    l = await store.aget_by_ids(ids)
    assert len(l) == 0


@pytest.mark.asyncio
async def test_asimilarity_search(store: CoherenceVectorStore):
    print()
    print(f"=======: {inspect.currentframe().f_code.co_name}")
    documents = get_test_data()
    await store.aadd_documents(documents)
    ids = [doc.id for doc in documents]
    l = await store.aget_by_ids(ids)
    assert len(l) == 10

    # result = await coherence_store.asimilarity_search("animal")
    result = await store.asimilarity_search("fruit")
    assert len(result) == 4
    print("====")
    for e in result:
        print(e)


@pytest.mark.asyncio
async def test_asimilarity_search_by_vector(store: CoherenceVectorStore):
    print()
    print(f"=======: {inspect.currentframe().f_code.co_name}")
    documents = get_test_data()
    await store.aadd_documents(documents)
    ids = [doc.id for doc in documents]
    l = await store.aget_by_ids(ids)
    assert len(l) == 10

    vector = store.embeddings.embed_query("animal")
    result = await store.asimilarity_search_by_vector(vector)
    assert len(result) == 4
    print("====")
    for e in result:
        print(e)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score(store: CoherenceVectorStore):
    print()
    print(f"=======: {inspect.currentframe().f_code.co_name}")
    documents = get_test_data()
    await store.aadd_documents(documents)
    ids = [doc.id for doc in documents]
    l = await store.aget_by_ids(ids)
    assert len(l) == 10

    # result = await coherence_store.asimilarity_search("animal")
    result = await store.asimilarity_search_with_score("fruit")
    assert len(result) == 4
    print("====")
    for e in result:
        print(e)


@pytest.mark.asyncio
async def test_afrom_texts():
    session = await Session.create()
    try:
        cache = await session.get_cache("test-map-async")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        texts = ["apple", "banana"]
        metadatas = [{"cat": "fruit"}, {"cat": "fruit"}]
        ids = ["id1", "id2"]

        cvs = await CoherenceVectorStore.afrom_texts(
            texts=texts,
            embedding=embedding,
            cache=cache,
            metadatas=metadatas,
            ids=ids,
        )

        results = await cvs.aget_by_ids(ids)
        assert len(results) == 2
    finally:
        await session.close()
