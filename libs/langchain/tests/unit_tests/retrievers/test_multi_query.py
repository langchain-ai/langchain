from langchain.embeddings import FakeEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.knn import KNNRetriever
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_multi_query() -> None:
    doc_list = [
        "I like apples",
        "I ate a banana",
        "Avocados and oranges are fruits",
    ]
    knn_retriever = KNNRetriever.from_texts(
        texts=doc_list, embeddings=FakeEmbeddings(size=100)
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=knn_retriever,
        llm=FakeLLM(),
    )

    queries = multi_query_retriever.generate_queries("I like apples")
    for q in queries:
        assert q == "foo"

    retrieved_docs = multi_query_retriever.retrieve_documents(queries)
    for doc in retrieved_docs:
        assert doc.page_content in doc_list

    union_docs = multi_query_retriever.unique_union(retrieved_docs)
    for doc in union_docs:
        assert doc.page_content in doc_list

    docs = multi_query_retriever.get_relevant_documents("I like apples")
    for doc in docs:
        assert doc.page_content in doc_list
