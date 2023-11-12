import pytest

from langchain.retrievers import MultiFieldRetriever, TFIDFRetriever
from langchain.schema.document import Document


@pytest.mark.requires("sklearn")
def test_multi_field_retriever() -> None:
    documents = [
        Document(
            page_content="Japanese Sake",
            metadata={"production year": 1991, "brand": "Yamazaki", "doc_id": 1},
        ),
        Document(
            page_content="Japanese Sake",
            metadata={"production year": 1981, "brand": "Dassai", "doc_id": 2},
        ),
        Document(page_content="Wine", metadata={"production year": 1920, "doc_id": 3}),
        Document(page_content="Wine", metadata={"production year": 1930, "doc_id": 4}),
    ]

    tfidf = TFIDFRetriever.from_documents(documents)

    tfidf_multi_field = MultiFieldRetriever.from_documents(
        retriever=TFIDFRetriever,
        documents=documents,
    )

    # the vanilla TFIDFRetriever should not return the correct document
    doc = tfidf.get_relevant_documents("Japanese Sake, Yamazaki")[0]
    assert doc.metadata["brand"] != "Yamazaki"

    # but with the MultiFieldRetriever it should
    doc = tfidf_multi_field.get_relevant_documents("Japanese Sake, Yamazaki")[0]
    assert doc.metadata["brand"] == "Yamazaki"
