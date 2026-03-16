from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.fake import FakeEmbeddings


def test_contract_clause_search():
    docs = [
        Document(page_content="Payment must be made within 30 days."),
        Document(page_content="Termination requires 60 days notice."),
    ]

    embeddings = FakeEmbeddings(size=32)

    vectorstore = FAISS.from_documents(docs, embeddings)

    results = vectorstore.similarity_search("termination")

    assert "Termination" in results[0].page_content
