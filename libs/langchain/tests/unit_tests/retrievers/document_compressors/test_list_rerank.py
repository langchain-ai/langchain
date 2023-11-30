import pytest

from langchain.retrievers.document_compressors.list_rerank import ListRerank
from langchain.schema import Document
from tests.unit_tests.llms.fake_llm import FakeLLM

query = "Do you have a pencil?"
top_n = 2
input_docs = [
    Document(page_content="I have a pen."),
    Document(page_content="Do you have a pen?"),
    Document(page_content="I have a bag."),
]


def test__list_rerank_success() -> None:
    llm = FakeLLM(
        queries={
            query: """
            ```json 
            {
                "reranked_documents": [
                    {"document_id": 1, "score": 0.99}, 
                    {"document_id": 0, "score": 0.95}, 
                    {"document_id": 2, "score": 0.50}
                ]
            }
            ```
            """
        },
        sequential_responses=True,
    )

    list_rerank = ListRerank.from_llm(llm=llm, top_n=top_n)
    output_docs = list_rerank.compress_documents(input_docs, query)

    assert len(output_docs) == top_n
    assert output_docs[0].metadata["relevance_score"] == 0.99
    assert output_docs[0].page_content == "Do you have a pen?"


def test__list_rerank_error() -> None:
    llm = FakeLLM(
        queries={
            query: """
            ```json 
            {
                "reranked_documents": [
                    {"<>": 1, "score": 0.99}, 
                    {"document_id": 0, "score": 0.95}, 
                    {"document_id": 2, "score": 0.50}
                ]
            }
            ```
            """
        },
        sequential_responses=True,
    )

    list_rerank = ListRerank.from_llm(llm=llm, top_n=top_n)

    with pytest.raises(KeyError) as excinfo:
        list_rerank.compress_documents(input_docs, query)
    assert "document_id" in str(excinfo.value)


def test__list_rerank_fallback() -> None:
    llm = FakeLLM(
        queries={
            query: """
            ```json 
            {
                "reranked_documents": [
                    {"<>": 1, "score": 0.99}, 
                    {"document_id": 0, "score": 0.95}, 
                    {"document_id": 2, "score": 0.50}
                ]
            }
            ```
            """
        },
        sequential_responses=True,
    )

    list_rerank = ListRerank.from_llm(llm=llm, top_n=top_n, fallback=True)
    output_docs = list_rerank.compress_documents(input_docs, query)
    assert len(output_docs) == top_n
