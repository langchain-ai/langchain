import pytest

from langchain.retrievers import LlamaIndexRetriever


@pytest.mark.requires("llama_index")
def test_LlamaIndexRetriever_vector_store_index() -> None:
    from llama_index import Document as LlamaIndexDocument
    from llama_index import VectorStoreIndex

    texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    documents = [LlamaIndexDocument(text=text) for text in texts]
    index = VectorStoreIndex.from_documents(documents)

    llama_index_retriever = LlamaIndexRetriever(index=index)
    got = llama_index_retriever.get_relevant_documents(query="I want a pen.")
    assert len(got) == 3


@pytest.mark.requires("llama_index")
def test_LlamaIndexRetriever_tree_index() -> None:
    from llama_index import Document as LlamaIndexDocument
    from llama_index import TreeIndex

    texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    documents = [LlamaIndexDocument(text=text) for text in texts]
    index = TreeIndex.from_documents(documents)

    llama_index_retriever = LlamaIndexRetriever(index=index)
    got = llama_index_retriever.get_relevant_documents(query="I want a pen.")
    assert len(got) == 3


@pytest.mark.requires("llama_index")
def test_LlamaIndexRetriever_list_index() -> None:
    from llama_index import Document as LlamaIndexDocument
    from llama_index import ListIndex

    texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    documents = [LlamaIndexDocument(text=text) for text in texts]
    index = ListIndex.from_documents(documents)

    llama_index_retriever = LlamaIndexRetriever(index=index)
    got = llama_index_retriever.get_relevant_documents(query="I want a pen.")
    assert len(got) == 3
