import faiss
import pytest

from langchain.docstore import InMemoryDocstore
from langchain.embeddings.fake import FakeEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores.faiss import FAISS

embedding_size = 1536

expected = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "vectorstore", "VectorStoreRetrieverMemory"],
    "kwargs": {},
    "repr": "",
    "obj": {
        "exclude_input_keys": [],
        "input_key": None,
        "memory_key": "history",
        "return_docs": False,
    },
}


@pytest.mark.requires("faiss-cpu")
def test_to_json() -> None:
    index = faiss.IndexFlatL2(embedding_size)
    embeddings = FakeEmbeddings(size=embedding_size)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    memory.save_context(
        {"input": "My favorite food is pizza"}, {"output": "that's good to know"}
    )
    memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
    memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})
    answer = memory.to_json()
    if "obj" in answer and "retriever" in answer["obj"]:
        del answer["obj"]["retriever"]
    assert answer == expected
