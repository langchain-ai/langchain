"""Test Google Cloud Document AI Warehouse retriever."""

import os

from langchain_core.documents import Document

from langchain_community.retrievers import GoogleDocumentAIWarehouseRetriever


def test_google_documentai_warehoure_retriever() -> None:
    """In order to run this test, you should provide a project_id and user_ldap.

    Example:
    export USER_LDAP=...
    export PROJECT_NUMBER=...
    """
    project_number = os.environ["PROJECT_NUMBER"]
    user_ldap = os.environ["USER_LDAP"]
    docai_wh_retriever = GoogleDocumentAIWarehouseRetriever(
        project_number=project_number
    )
    documents = docai_wh_retriever.invoke(
        "What are Alphabet's Other Bets?", user_ldap=user_ldap
    )
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
