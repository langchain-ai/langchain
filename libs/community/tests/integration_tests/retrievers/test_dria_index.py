import pytest
from langchain_core.documents import Document

from langchain_community.retrievers import DriaRetriever


# Set a fixture for DriaRetriever
@pytest.fixture
def dria_retriever() -> DriaRetriever:
    api_key = "<YOUR_API_KEY>"
    contract_id = "B16z9i3rRi0KEeibrzzMU33YTB4WDtos1vdiMBTmKgs"
    retriever = DriaRetriever(api_key=api_key, contract_id=contract_id)
    return retriever


def test_dria_retriever(dria_retriever: DriaRetriever) -> None:
    texts = [
        {
            "text": "Langchain",
            "metadata": {
                "source": "source#1",
                "document_id": "doc123",
                "content": "Langchain",
            },
        }
    ]
    dria_retriever.add_texts(texts)

    # Assuming get_relevant_documents returns a list of Document instances
    docs = dria_retriever.get_relevant_documents("Langchain")

    # Perform assertions
    assert len(docs) > 0, "Expected at least one document"
    doc = docs[0]
    assert isinstance(doc, Document), "Expected a Document instance"
    assert isinstance(doc.page_content, str), (
        "Expected document content type " "to be string"
    )
    assert isinstance(
        doc.metadata, dict
    ), "Expected document metadata content to be a dictionary"
