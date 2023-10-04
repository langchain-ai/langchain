"""Test Google Cloud DocAI Warehouse retriever."""
import os

from langchain.retrievers import GoogleDocaiWarehouseSearchRetriever
from langchain.schema import Document


def test_google_docai_warehoure_retriever() -> None:
    """In order to run this test, you should provide a project_id and user_ldap.

    Example:
    export USER_LDAP=...
    export PROJECT=...
    """
    project_id = os.environ["PROJECT_ID"]
    user_ldap = os.environ["USER_LDAP"]
    docai_wh_retriever = GoogleDocaiWarehouseSearchRetriever(project_id=project_id)
    documents = docai_wh_retriever.get_relevant_documents(
        "What are Alphabet's Other Bets?", user_ldap=user_ldap
    )
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
